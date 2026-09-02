"""
Unified Target-Decoy Analysis Processor

Cross-run (2-fold CV) PmSM scoring. PmSM assignment and FDR control are
separate concerns handled by the caller -- see
:func:`~delpi.search.pmsm_assignment.assign_pmsms_across_runs` and
:class:`~delpi.search.tda.fdr_analyzer.FDRAnalyzer`, both run on the
DataFrame returned by :meth:`TDAProcessor.run_global`.
"""

import logging
from pathlib import Path
from typing import Callable, Literal, Tuple

SplitLevel = Literal["pmsm", "precursor", "peptide"]
GroupingType = Literal["lead_only", "parsimonious_grouping"]

import numpy as np
import polars as pl
import torch
from torch.utils.data import TensorDataset

from delpi.database.peptide_database import PeptideDatabase
from delpi.search.result_aggregator import ResultsAggregator
from delpi.search.result_manager import ResultManager
from delpi.search.tda.trainer import TargetDecoyTrainer
from delpi.constants import (
    DEFAULT_Q_VALUE_CUTOFF,
    TDA_MAX_TRAIN_SIZE,
    PMSM_EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)

FEATURE_DIM = PMSM_EMBEDDING_DIM + 1  # embedding + RT difference
RT_SCALE = 1000.0

# Type alias: given a DataFrame subset, return (n, FEATURE_DIM) numpy array
FeatureLoader = Callable[[pl.DataFrame], np.ndarray]


class TDAProcessor:
    """Unified target-decoy analysis for single-run and cross-run modes."""

    def __init__(
        self,
        db_dir: Path,
        output_dir: Path,
        device: torch.device,
        q_value_cutoff: float = DEFAULT_Q_VALUE_CUTOFF,
        use_protein_picker: bool = True,
        grouping_type: GroupingType = "parsimonious_grouping",
        n_ensemble: int = 1,
        ensemble_train_ratio: float = 0.8,
        batch_size: int = 2048,
        split_level: SplitLevel = "peptide",
    ):
        self.db_dir = db_dir
        self.output_dir = output_dir
        self.device = device
        self.q_value_cutoff = q_value_cutoff
        self.use_protein_picker = use_protein_picker
        self.grouping_type = grouping_type
        self.n_ensemble = n_ensemble
        self.ensemble_train_ratio = ensemble_train_ratio
        self.batch_size = batch_size
        self.split_level = split_level

    # ==================================================================
    # Public entry points
    # ==================================================================

    def run_global(
        self,
        result_aggregator: ResultsAggregator,
        group_key: str,
        training_params: dict = None,
        pass_label: str = "global",
    ) -> pl.DataFrame:
        """Cross-run TDA across multiple LC-MS runs (2-fold CV).

        Split granularity is controlled by ``self.split_level``.
        Features are loaded on-demand per subset.  Training data in each
        fold is subsampled when it exceeds ``TDA_MAX_TRAIN_SIZE``.

        When ``n_ensemble > 1``, each fold trains K models via bootstrap
        sampling (each of size ``ensemble_train_ratio`` × N_fold) with
        different seeds, and averages their logits.

        ``pass_label`` (e.g. ``"first"``/``"second"``) is used as the prefix
        for saved model artifacts/logs (``{pass_label}_tda_{fold}``) so that
        first- and second-pass training runs don't collide/overwrite each
        other and are easy to tell apart on disk.

        Returns every scored PmSM (joined with `frame_num`/`predicted_rt`/
        `observed_rt`/`is_decoy`/etc.) -- **not yet** assigned or
        FDR-annotated; the caller is expected to run
        :func:`~delpi.search.pmsm_assignment.assign_pmsms_across_runs` and
        then :class:`~delpi.search.tda.fdr_analyzer.FDRAnalyzer` on the result.
        """
        pmsm_df = self._load_multi_run(result_aggregator, group_key)
        feature_fn = self._make_aggregator_feature_fn(result_aggregator, group_key)

        fold_a, fold_b = self._split_pmsm_df(pmsm_df, level=self.split_level)
        full_pmsm_df = pmsm_df  # keep reference for join-back after scoring

        model_version_prefix = f"{pass_label}_tda"

        # Fold A trains → score Fold B
        scores_b = self._train_and_score_fold(
            fold_a,
            fold_b,
            feature_fn,
            fold_label="f0",
            training_params=training_params,
            model_version_prefix=model_version_prefix,
        )
        # Fold B trains → score Fold A
        scores_a = self._train_and_score_fold(
            fold_b,
            fold_a,
            feature_fn,
            fold_label="f1",
            training_params=training_params,
            model_version_prefix=model_version_prefix,
        )

        # Merge scored folds
        scored_a = fold_a.with_columns(pl.Series(values=scores_a, name="score")).select(
            "run_index", "pmsm_index", "cluster", "score"
        )
        scored_b = fold_b.with_columns(pl.Series(values=scores_b, name="score")).select(
            "run_index", "pmsm_index", "cluster", "score"
        )
        scored_df = pl.concat([scored_a, scored_b], how="vertical")

        # for debugging: save all scored PmSMs before selection
        scored_df.write_parquet(self.output_dir / "pmsm_scores.parquet")
        # scored_df = pl.read_parquet(self.output_dir / "pmsm_scores.parquet")

        # Bring in the rest of each PmSM's columns (frame_num, predicted_rt,
        # observed_rt, is_decoy, ...) so the returned DataFrame is
        # self-contained for the assignment/FDR steps that follow.
        scored_df = scored_df.join(
            full_pmsm_df.select(pl.exclude("cluster", "score")),
            on=["run_index", "pmsm_index"],
            how="left",
        )
        return scored_df

    def write_back_scores(
        self,
        pmsm_df: pl.DataFrame,
        result_aggregator: ResultsAggregator,
        group_key: str,
    ) -> None:
        """Persist ``score`` and run-specific ``precursor_q_value`` from a
        cross-run (:meth:`run_global`) result back into each run's own HDF
        group, aligned to that run's original pmsm_index order.

        Used so that later per-run steps (e.g. RT calibration ahead of a
        following full search) can bootstrap from these results without a
        redundant run-specific TDA pass.
        """
        for run_index, result_manager in result_aggregator._results_dict.items():
            sub_df = pmsm_df.filter(pl.col("run_index") == run_index)
            if sub_df.shape[0] == 0:
                continue

            num_raw = result_manager.read_dict(
                group_key, data_keys=["precursor_index"]
            )["precursor_index"].shape[0]
            score_arr = np.full(num_raw, np.nan, dtype=np.float32)
            q_value_arr = np.full(num_raw, np.nan, dtype=np.float32)
            idx = sub_df["pmsm_index"].to_numpy()
            score_arr[idx] = sub_df["score"].to_numpy()
            q_value_arr[idx] = sub_df["precursor_q_value"].to_numpy()

            result_manager.write_dict(
                group_key, {"score": score_arr, "precursor_q_value": q_value_arr}
            )

    def _train_and_score_fold(
        self,
        train_fold: pl.DataFrame,
        test_fold: pl.DataFrame,
        feature_fn: FeatureLoader,
        fold_label: str,
        training_params: dict = None,
        model_version_prefix: str = "global_tda",
    ) -> np.ndarray:
        """Train on *train_fold*, score *test_fold*.

        When ``n_ensemble > 1``, bootstraps K models from *train_fold*
        (after subsampling) and averages their logits.
        """
        train_df = self._subsample_train(train_fold)

        if self.n_ensemble <= 1:
            train_dataset = self._build_tensor_dataset(train_df, feature_fn)
            model = self._train_model(
                train_dataset,
                model_version=f"{model_version_prefix}_{fold_label}",
                training_params=training_params,
            )
            return self._score(test_fold, model, feature_fn)

        return self._ensemble_score(
            train_df,
            test_fold,
            feature_fn,
            fold_label=fold_label,
            training_params=training_params,
            model_version_prefix=model_version_prefix,
        )

    # ==================================================================
    # Data loading
    # ==================================================================

    @staticmethod
    def _load_multi_run(
        result_aggregator: ResultsAggregator,
        group_key: str,
    ) -> pl.DataFrame:
        """Load PmSM data (without features) for all runs, attach is_decoy."""
        pmsm_df = result_aggregator.load_pmsm_df(group_key=group_key)
        pmsm_df = PeptideDatabase.join(
            result_aggregator.db_dir,
            pmsm_df,
            precursor_columns=[],
            modification_columns=[],
            peptide_columns=["is_decoy"],
        )
        return pmsm_df

    # ==================================================================
    # Feature loading strategies
    # ==================================================================

    @staticmethod
    def _make_aggregator_feature_fn(
        result_aggregator: ResultsAggregator,
        group_key: str,
    ) -> FeatureLoader:
        """Feature loader that reads from HDF files via ResultsAggregator."""

        def _load(df: pl.DataFrame) -> np.ndarray:
            arr = result_aggregator.load_features(
                df, group_key=group_key, feature_dim=FEATURE_DIM
            )
            arr[:, -1] = (df["observed_rt"] - df["predicted_rt"]).to_numpy() / RT_SCALE
            return arr

        return _load

    # ==================================================================
    # Shared pipeline steps
    # ==================================================================

    @staticmethod
    def _split_pmsm_df(
        pmsm_df: pl.DataFrame,
        level: SplitLevel = "pmsm",
        seed: int = 42,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """2-fold split of PmSMs at the requested granularity.

        Parameters
        ----------
        level
            ``"pmsm"`` (default): random row-level split.  Same peptide/
            precursor can appear in both folds.  Cheapest but allows mild
            information leakage through shared sequence embeddings.
            ``"precursor"``: split on ``precursor_index`` so that every
            (peptide, charge, mod) variant is confined to one fold.
            ``"peptide"``: split on ``peptide_index`` so that every
            sequence is confined to one fold (most conservative, à la
            Percolator).
        seed
            RNG seed for the shuffle.
        """
        if level == "pmsm":
            shuffled = pmsm_df.sample(
                fraction=1.0, with_replacement=False, shuffle=True, seed=seed
            )
            mid = shuffled.shape[0] // 2
            return shuffled.head(mid), shuffled.slice(mid)

        if level == "precursor":
            group_col = "precursor_index"
        elif level == "peptide":
            group_col = "peptide_index"
        else:
            raise ValueError(
                f"Unknown split level: {level!r}. "
                "Expected one of 'pmsm', 'precursor', 'peptide'."
            )

        # Sort before shuffle: unique() returns rows in a non-deterministic
        # order, so we canonicalise first to make the seeded shuffle reproducible.
        group_ids = (
            pmsm_df.select(group_col)
            .unique()
            .sort(group_col)[group_col]
            .shuffle(seed=seed)
        )
        mid = len(group_ids) // 2
        fold_a_ids = group_ids[:mid].to_frame()
        fold_b_ids = group_ids[mid:].to_frame()

        fold_a_df = pmsm_df.join(
            fold_a_ids, on=group_col, how="inner", maintain_order="left"
        )
        fold_b_df = pmsm_df.join(
            fold_b_ids, on=group_col, how="inner", maintain_order="left"
        )

        return fold_a_df, fold_b_df

    @staticmethod
    def _subsample_train(train_df: pl.DataFrame) -> pl.DataFrame:
        """Subsample high-scoring PmSMs per precursor when dataset is too large."""
        if train_df.shape[0] > TDA_MAX_TRAIN_SIZE:
            train_df = train_df.sample(
                n=TDA_MAX_TRAIN_SIZE, with_replacement=False, shuffle=True, seed=1221
            )
        return train_df.sort(["run_index", "pmsm_index"])

    @staticmethod
    def _build_tensor_dataset(
        df: pl.DataFrame,
        feature_fn: FeatureLoader,
    ) -> TensorDataset:
        """Build a TensorDataset from a subset DataFrame."""
        feature_arr = feature_fn(df)
        x = torch.from_numpy(feature_arr)
        y = (~df["is_decoy"].to_torch()).float().unsqueeze(1)
        return TensorDataset(x, y)

    def _train_model(
        self,
        train_dataset: TensorDataset,
        model_version: str,
        training_params: dict = None,
        model_params: dict = None,
        seed: int = None,
    ):
        """Train the TDA classifier and return the best model on self.device.

        ``training_params`` is forwarded to :class:`TargetDecoyTrainer`.
        When ``seed`` is provided, it overrides ``random_seed`` in the dict.
        """
        training_params = dict(training_params) if training_params else {}
        if seed is not None:
            training_params["random_seed"] = seed

        trainer = TargetDecoyTrainer(
            model_params=model_params, training_params=training_params
        )
        trainer.train(
            model_version=model_version,
            train_dataset=train_dataset,
            output_dir=self.output_dir,
            device=self.device,
        )
        return trainer.get_best_model().to(self.device).eval()

    def _score(
        self,
        df: pl.DataFrame,
        model: torch.nn.Module,
        feature_fn: FeatureLoader,
    ) -> np.ndarray:
        """Score a subset of PmSMs with a trained model."""
        feature_arr = feature_fn(df)
        return self._batched_inference(model, feature_arr, self.batch_size)

    def _ensemble_score(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        feature_fn: FeatureLoader,
        fold_label: str = "",
        training_params: dict = None,
        model_version_prefix: str = "global_tda",
    ) -> np.ndarray:
        """Train K models via bootstrap from *train_df* and average logits."""
        test_feature_arr = feature_fn(test_df)
        n_train = len(train_df)
        subset_size = int(n_train * self.ensemble_train_ratio)
        avg_scores = np.zeros(len(test_df), dtype=np.float64)

        for k in range(self.n_ensemble):
            seed = 42 + k
            indices = np.random.RandomState(seed).choice(
                n_train, size=subset_size, replace=False
            )
            subset_df = train_df[indices.tolist()]
            train_dataset = self._build_tensor_dataset(subset_df, feature_fn)

            model = self._train_model(
                train_dataset,
                model_version=f"{model_version_prefix}_{fold_label}_e{k}",
                training_params=training_params,
                seed=seed,
            )
            scores = self._batched_inference(model, test_feature_arr, self.batch_size)
            avg_scores += scores
            logger.info(
                f"Ensemble {fold_label} model {k + 1}/{self.n_ensemble} trained"
            )

        avg_scores /= self.n_ensemble
        return avg_scores.astype(np.float32)

    @staticmethod
    def _batched_inference(
        model: torch.nn.Module,
        feature_arr: np.ndarray,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Run batched GPU inference without DataLoader overhead."""
        device = next(model.parameters()).device
        x_all = torch.from_numpy(feature_arr)
        scores = np.empty(len(feature_arr), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(x_all), batch_size):
                end = min(start + batch_size, len(x_all))
                logits = model(x_all[start:end].to(device))
                scores[start:end] = logits.flatten().cpu().numpy()
        return scores
