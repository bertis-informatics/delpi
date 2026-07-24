"""
Unified Target-Decoy Analysis Processor

Supports both run-specific (single-run) and global (cross-run) TDA pipelines.
Common logic (split, build dataset, train, score, cluster selection) is shared;
the two entry points differ only in data loading, cluster grouping, and FDR scope.
"""

import logging
from pathlib import Path
from typing import Callable, List, Literal, Tuple

SplitLevel = Literal["pmsm", "precursor", "peptide"]
GroupingType = Literal["lead_only", "parsimonious_grouping"]

import numpy as np
import polars as pl
import torch
from torch.utils.data import TensorDataset

from delpi.database.peptide_database import PeptideDatabase
from delpi.search.result_aggregator import ResultsAggregator
from delpi.search.result_manager import ResultManager
from delpi.search.tda.fdr_analyzer import FDRAnalyzer
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

    def run_single(
        self,
        result_manager: ResultManager,
        group_key: str,
        training_params: dict = None,
    ) -> pl.DataFrame:
        """Run-specific TDA for a single LC-MS run (2-fold CV).

        Split granularity is controlled by ``self.split_level``.
        All features fit in memory, so no subsampling is applied.

        Writes ``score`` and ``precursor_q_value`` arrays back into *group_key*
        inside *result_manager*, aligned to the original HDF pmsm_index order.
        Entries that did not survive cluster/precursor deduplication are NaN.

        Returns
        -------
        pl.DataFrame
            Best-scoring PmSMs (one per precursor) with ``score`` and
            ``precursor_q_value`` columns.
        """
        pmsm_df, feature_arr = self._load_single_run(
            result_manager, group_key, self.db_dir
        )

        num_decoys = pmsm_df["is_decoy"].sum()
        num_targets = pmsm_df.shape[0] - num_decoys
        logger.info(
            f"Training a classifier with {num_targets:,} positive and {num_decoys:,} negative PmSMs"
        )

        feature_fn = self._make_array_feature_fn(feature_arr)

        fold_a, fold_b = self._split_pmsm_df(pmsm_df, level=self.split_level)

        # Fold A trains → score Fold B
        ds_a = self._build_tensor_dataset(fold_a, feature_fn)
        model_a = self._train_model(
            ds_a,
            model_version=f"{result_manager.run_name}_f0",
            training_params=training_params,
        )
        scores_b = self._score(fold_b, model_a, feature_fn)

        # Fold B trains → score Fold A
        ds_b = self._build_tensor_dataset(fold_b, feature_fn)
        model_b = self._train_model(
            ds_b,
            model_version=f"{result_manager.run_name}_f1",
            training_params=training_params,
        )
        scores_a = self._score(fold_a, model_b, feature_fn)

        # Merge scored folds – every PmSM now has a score
        scored_a = fold_a.with_columns(pl.Series(values=scores_a, name="score"))
        scored_b = fold_b.with_columns(pl.Series(values=scores_b, name="score"))
        full_scored_df = pl.concat([scored_a, scored_b], how="vertical")

        # Best per cluster, then best per precursor.
        # Tie-break on pmsm_index to make selection deterministic when
        # multiple PmSMs in a group share the same score.
        pmsm_df = full_scored_df.group_by("cluster").agg(
            pl.all().sort_by(["score", "pmsm_index"]).last()
        )

        fdr = FDRAnalyzer(
            q_value_cutoff=self.q_value_cutoff,
            db_dir=self.db_dir,
            use_protein_picker=self.use_protein_picker,
            grouping_type=self.grouping_type,
        )
        pmsm_df = fdr.perform_run_specific_analysis(pmsm_df)

        q_value_arr = np.full(feature_arr.shape[0], np.nan, dtype=np.float32)
        score_arr = np.full(feature_arr.shape[0], np.nan, dtype=np.float32)
        score_arr[full_scored_df["pmsm_index"]] = full_scored_df["score"].to_numpy()
        q_value_arr[pmsm_df["pmsm_index"]] = pmsm_df["precursor_q_value"].to_numpy()

        result_manager.write_dict(
            group_key, {"score": score_arr, "precursor_q_value": q_value_arr}
        )

        return pmsm_df

    def run_global(
        self,
        result_aggregator: ResultsAggregator,
        group_key: str,
        training_params: dict = None,
        run_protein_grouping: bool = True,
        library_confidence_df: pl.DataFrame = None,
        pass_label: str = "global",
    ) -> pl.DataFrame:
        """Cross-run TDA across multiple LC-MS runs (2-fold CV).

        Split granularity is controlled by ``self.split_level``.
        Features are loaded on-demand per subset.  Training data in each
        fold is subsampled when it exceeds ``TDA_MAX_TRAIN_SIZE``.

        When ``n_ensemble > 1``, each fold trains K models via bootstrap
        sampling (each of size ``ensemble_train_ratio`` × N_fold) with
        different seeds, and averages their logits.

        When ``library_confidence_df`` is provided (second pass of the
        two-pass MBR search), protein-group membership is reused from the
        first pass for targets and freshly computed for decoys — see
        :meth:`FDRAnalyzer.perform_global_analysis`. This takes precedence
        over ``run_protein_grouping``, which otherwise controls whether
        protein inference is performed from scratch (first pass) or skipped
        entirely.

        ``pass_label`` (e.g. ``"first"``/``"second"``) is used as the prefix
        for saved model artifacts/logs (``{pass_label}_tda_{fold}``) so that
        first- and second-pass training runs don't collide/overwrite each
        other and are easy to tell apart on disk.

        Returns an annotated PmSM DataFrame with global and run-specific
        q-values.
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

        # Merge scored folds, then select best per (run, cluster)
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

        pmsm_df = scored_df.group_by(["run_index", "cluster"]).agg(
            pl.all().sort_by(["score", "pmsm_index"]).last()
        )
        pmsm_df = pmsm_df.join(
            full_pmsm_df.select(pl.exclude("cluster")),
            how="left",
            on=["run_index", "pmsm_index"],
        ).drop("cluster")

        logger.info(
            f"Selected {pmsm_df.shape[0]} non-redundant PmSMs "
            f"(one per run/cluster) from {len(scored_df)} scored PmSMs"
        )

        pmsm_df = self._join_database_columns(pmsm_df, result_aggregator.db_dir)

        fdr = FDRAnalyzer(
            q_value_cutoff=self.q_value_cutoff,
            db_dir=self.db_dir,
            use_protein_picker=self.use_protein_picker,
            grouping_type=self.grouping_type,
        )
        pmsm_df = fdr.perform_global_analysis(
            pmsm_df,
            protein_inference=run_protein_grouping,
            library_confidence_df=library_confidence_df,
        )
        pmsm_df = fdr.batch_run_specific_analysis(pmsm_df)
        pmsm_df = pmsm_df.join(
            result_aggregator.get_run_df(), on="run_index", how="left"
        )
        return pmsm_df

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
    def _load_single_run(
        result_manager: ResultManager,
        group_key: str,
        db_dir: Path,
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """Load PmSM data and features for a single run."""
        results_dict = result_manager.read_dict(
            group_key,
            data_keys=[
                "precursor_index",
                "frame_num",
                "cluster",
                "predicted_rt",
                "observed_rt",
                "logit",
            ],
        )
        pmsm_df = pl.DataFrame(results_dict).with_row_index("pmsm_index")
        pmsm_df = PeptideDatabase.join(
            db_dir,
            pmsm_df,
            precursor_columns=["precursor_charge"],
            modification_columns=["mod_ids", "mod_sites"],
            peptide_columns=[
                "peptide",
                "sequence_length",
                "is_decoy",
                "protein_index",
            ],
        )

        # Read embeddings directly into (N, FEATURE_DIM) array; fill RT diff
        feature_arr = result_manager.load_features(group_key, feature_dim=FEATURE_DIM)
        feature_arr[:, -1] = (
            pmsm_df["observed_rt"] - pmsm_df["predicted_rt"]
        ).to_numpy() / RT_SCALE
        return pmsm_df, feature_arr

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
    def _make_array_feature_fn(all_features: np.ndarray) -> FeatureLoader:
        """Feature loader that indexes a pre-built (N, FEATURE_DIM) array."""

        def _load(df: pl.DataFrame) -> np.ndarray:
            return all_features[df["pmsm_index"]]

        return _load

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
        seed: int = None,
    ):
        """Train the TDA classifier and return the best model on self.device.

        ``training_params`` is forwarded to :class:`TargetDecoyTrainer`.
        When ``seed`` is provided, it overrides ``random_seed`` in the dict.
        """
        training_params = dict(training_params) if training_params else {}
        if seed is not None:
            training_params["random_seed"] = seed

        trainer = TargetDecoyTrainer(training_params=training_params)
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

    @staticmethod
    def _select_best(
        test_df: pl.DataFrame,
        score_arr: np.ndarray,
        group_keys: List[str],
        full_pmsm_df: pl.DataFrame = None,
    ) -> pl.DataFrame:
        """Keep the highest-scoring PmSM per group (cluster or run+cluster).

        When *full_pmsm_df* is provided (global mode), the result is joined
        back to the full DataFrame to recover all columns.
        """
        scored = test_df.with_columns(pl.Series(values=score_arr, name="score"))

        if full_pmsm_df is not None:
            # Global: select minimal columns, join back after grouping
            scored = scored.select("run_index", "pmsm_index", "cluster", "score")

        pmsm_df = scored.group_by(group_keys).agg(
            pl.all().sort_by(["score", "pmsm_index"]).last()
        )

        if full_pmsm_df is not None:
            pmsm_df = pmsm_df.join(
                full_pmsm_df.select(pl.exclude("cluster")),
                how="left",
                on=["run_index", "pmsm_index"],
            ).drop("cluster")

        logger.info(
            f"Selected {pmsm_df.shape[0]} non-redundant PmSMs "
            f"(one per {'/'.join(group_keys)}) from {len(score_arr)} PmSMs"
        )
        return pmsm_df

    @staticmethod
    def _join_database_columns(
        pmsm_df: pl.DataFrame,
        db_dir: Path,
    ) -> pl.DataFrame:
        """Attach precursor/modification/peptide/fasta_id columns from the database."""
        # Columns added by the join — drop them first if already present to
        # avoid polars creating duplicate (_right) columns.
        _join_added = {
            "peptidoform_index",
            "peptide_index",
            "precursor_charge",
            "mod_ids",
            "mod_sites",
            "peptide",
            "sequence_length",
            "is_decoy",
            "protein_index",
            "fasta_id",
        }
        cols_to_drop = [c for c in pmsm_df.columns if c in _join_added]
        pmsm_df = PeptideDatabase.join(
            db_dir,
            pmsm_df.drop(cols_to_drop),
            precursor_columns=["precursor_charge"],
            modification_columns=["mod_ids", "mod_sites"],
            peptide_columns=[
                "peptide",
                "sequence_length",
                "is_decoy",
                "protein_index",
            ],
        )

        # Attach a semicolon-joined `fasta_id` column per precursor, derived
        # from the protein sequence database (was previously done via
        # FDRAnalyzer.add_fasta_id_column, now folded in here so all DB joins
        # live in one place).
        fasta_id_df = (
            pl.scan_parquet(db_dir / "sequence_df.parquet")
            .select(pl.col("protein_index", "fasta_id"))
            .collect()
        )
        fasta_lookup_df = (
            pmsm_df.select(pl.col("precursor_index", "protein_index"))
            .unique(subset="precursor_index", keep="first")
            .explode("protein_index")
            .join(fasta_id_df, on="protein_index", how="left")
            .group_by("precursor_index")
            .agg(pl.col("fasta_id"))
            .with_columns(pl.col("fasta_id").list.sort().list.join(";"))
        )

        return pmsm_df.join(fasta_lookup_df, on="precursor_index", how="left")
