"""
Unified Target-Decoy Analysis Processor

Cross-run (2-fold CV) PmSM scoring. PmSM assignment and FDR control are
separate concerns handled by the caller -- see
:func:`~delpi.search.pmsm_assignment.assign_pmsms_across_runs` and
:meth:`~delpi.search.search_manager.SearchManager._infer_proteins_and_analyze_fdr`,
which orchestrates protein inference, group scoring and FDR on the
DataFrame returned by :meth:`TDAProcessor.run_global`.
"""

import logging
from pathlib import Path
from typing import Literal, Tuple

SplitLevel = Literal["pmsm", "precursor", "peptide"]
GroupingType = Literal["lead_only", "parsimonious_grouping"]

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from delpi.database.peptide_database import PeptideDatabase
from delpi.search.result_aggregator import ResultsAggregator
from delpi.search.result_manager import ResultManager
from delpi.search.tda.features import FEATURE_DIM, compute_rescoring_features
from delpi.search.tda.pmsm_features import FEATURE_ROW, PmsmFeatures
from delpi.search.tda.trainer import DEFAULT_TRAINING_PARAMS, TargetDecoyTrainer
from delpi.constants import (
    DEFAULT_Q_VALUE_CUTOFF,
    TDA_MAX_FEATURE_MEMORY,
    TDA_MAX_TRAIN_SIZE,
)

logger = logging.getLogger(__name__)

# Library columns joined only to compute rescoring features; not returned.
_FEATURE_INPUT_COLUMNS = ("precursor_mz", "sequence_length")


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
        pass_label: str = "first",
        model_params: dict = None,
    ) -> pl.DataFrame:
        """Cross-run TDA across multiple LC-MS runs (2-fold CV).

        Split granularity is controlled by ``self.split_level``.  Training
        data in each fold is subsampled when it exceeds ``TDA_MAX_TRAIN_SIZE``.
        Both fold models are trained first; then each run is scored, every
        PmSM by the model(s) of the other fold.

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
        then infer/score proteins and estimate FDR via
        :meth:`~delpi.search.search_manager.SearchManager._infer_proteins_and_analyze_fdr`.
        """
        pmsm_df = self._load_multi_run(result_aggregator, group_key)
        fold_a, fold_b = self._split_pmsm_df(pmsm_df, level=self.split_level)
        features = PmsmFeatures(
            result_aggregator,
            group_key,
            pmsm_df,
            self._scaled_rescoring_features(result_aggregator, pmsm_df),
            in_memory=self._features_fit_in_memory(pmsm_df, fold_a, fold_b),
        )

        models = [
            self._train_fold_models(
                fold,
                features,
                fold_label=f"f{k}",
                training_params=training_params,
                model_params=model_params,
                model_version_prefix=f"{pass_label}_tda",
            )
            for k, fold in enumerate((fold_a, fold_b))
        ]
        in_fold_b = np.zeros(pmsm_df.height, dtype=bool)
        in_fold_b[fold_b[FEATURE_ROW].to_numpy()] = True
        scores = self._score_out_of_fold(features, in_fold_b, models)

        return pmsm_df.with_columns(pl.Series("score", scores)).select(
            "run_index",
            "pmsm_index",
            "cluster",
            "score",
            pl.exclude(
                "run_index",
                "pmsm_index",
                "cluster",
                "score",
                FEATURE_ROW,
                *_FEATURE_INPUT_COLUMNS,
            ),
        )

    def _train_fold_models(
        self,
        train_fold: pl.DataFrame,
        features: PmsmFeatures,
        fold_label: str,
        training_params: dict = None,
        model_version_prefix: str = "global_tda",
        model_params: dict = None,
    ) -> list[torch.nn.Module]:
        """Train the model(s) of one fold. With ``n_ensemble > 1``, K models
        on bootstrap subsets of *train_fold* (after subsampling)."""
        train_df = self._subsample_train(train_fold, level=self.split_level)

        if self.n_ensemble <= 1:
            # (subset, split seed, model version, trainer seed)
            runs = [
                (
                    train_df,
                    self._training_param(training_params, "random_seed"),
                    f"{model_version_prefix}_{fold_label}",
                    None,
                )
            ]
        else:
            subset_size = int(len(train_df) * self.ensemble_train_ratio)
            runs = []
            for k in range(self.n_ensemble):
                seed = 42 + k
                indices = np.random.RandomState(seed).choice(
                    len(train_df), size=subset_size, replace=False
                )
                runs.append(
                    (
                        train_df[indices.tolist()],
                        seed,
                        f"{model_version_prefix}_{fold_label}_e{k}",
                        seed,
                    )
                )

        models = []
        for subset_df, split_seed, model_version, seed in runs:
            fit_df, val_df = self._split_train_val(
                subset_df,
                level=self.split_level,
                train_frac=self._training_param(training_params, "train_split"),
                max_val_samples=self._training_param(
                    training_params, "max_val_samples"
                ),
                seed=split_seed,
            )
            models.append(
                self._train_model(
                    train_dataset=self._build_tensor_dataset(fit_df, features),
                    val_dataset=self._build_tensor_dataset(val_df, features),
                    model_version=model_version,
                    training_params=training_params,
                    model_params=model_params,
                    seed=seed,
                )
            )
        return models

    def _score_out_of_fold(
        self,
        features: PmsmFeatures,
        in_fold_b: np.ndarray,
        models: list[list[torch.nn.Module]],
    ) -> np.ndarray:
        """Score each PmSM with the models of the other fold (`models[0]`:
        fold A, `models[1]`: fold B), averaging ensemble logits."""
        models_a, models_b = models
        scores = np.empty(len(in_fold_b), dtype=np.float32)
        for start, end, inputs in features.run_blocks():
            logits = self._batched_inference(models_a + models_b, inputs, self.batch_size)
            by_models_a = logits[: len(models_a)].mean(axis=0)
            by_models_b = logits[len(models_a) :].mean(axis=0)
            scores[start:end] = np.where(in_fold_b[start:end], by_models_a, by_models_b)
        return scores

    # ==================================================================
    # Data loading
    # ==================================================================

    @staticmethod
    def _load_multi_run(
        result_aggregator: ResultsAggregator,
        group_key: str,
    ) -> pl.DataFrame:
        """Load PmSM data (without embeddings) for all runs, attach is_decoy
        and the library columns used by the rescoring features. Rows are
        sorted by (run_index, pmsm_index); ``FEATURE_ROW`` is the row number.
        """
        pmsm_df = result_aggregator.load_pmsm_df(group_key=group_key)
        pmsm_df = PeptideDatabase.join(
            result_aggregator.db_dir,
            pmsm_df,
            precursor_columns=["precursor_mz"],
            modification_columns=[],
            peptide_columns=["is_decoy", "sequence_length"],
        )
        return pmsm_df.sort(["run_index", "pmsm_index"]).with_row_index(FEATURE_ROW)

    @staticmethod
    def _scaled_rescoring_features(
        result_aggregator: ResultsAggregator, pmsm_df: pl.DataFrame
    ) -> np.ndarray:
        """Rescoring features z-scored over all PmSMs (label-free, so no
        leakage between folds)."""
        gradient_lengths = TDAProcessor._gradient_lengths(result_aggregator, pmsm_df)
        rescoring_features = compute_rescoring_features(pmsm_df, gradient_lengths)
        scaler = StandardScaler().fit(rescoring_features)
        logger.debug(
            "Rescoring feature mean %s, scale %s", scaler.mean_, scaler.scale_
        )
        return scaler.transform(rescoring_features).astype(np.float32)

    @staticmethod
    def _features_fit_in_memory(
        pmsm_df: pl.DataFrame, fold_a: pl.DataFrame, fold_b: pl.DataFrame
    ) -> bool:
        """Whether all PmSMs' inputs plus one fold's training copy fit in
        ``TDA_MAX_FEATURE_MEMORY``."""
        n_train = min(max(fold_a.height, fold_b.height), TDA_MAX_TRAIN_SIZE)
        row_bytes = FEATURE_DIM * np.dtype(np.float32).itemsize
        needed = (pmsm_df.height + n_train) * row_bytes
        in_memory = needed <= TDA_MAX_FEATURE_MEMORY
        logger.info(
            f"Rescoring inputs of {pmsm_df.height:,} PmSMs: "
            f"{'kept in memory' if in_memory else 'read run by run'} "
            f"({needed / 2**30:.1f} GiB needed in memory, "
            f"limit {TDA_MAX_FEATURE_MEMORY / 2**30:.1f} GiB)"
        )
        return in_memory

    @staticmethod
    def _gradient_lengths(
        result_aggregator: ResultsAggregator, pmsm_df: pl.DataFrame
    ) -> dict[int, float]:
        """Gradient length (seconds) of every run in `pmsm_df`, from the
        result file; falls back to the run's latest observed RT."""
        lengths = {}
        for run_index in pmsm_df["run_index"].unique().sort().to_list():
            try:
                value = float(
                    result_aggregator.get_result_manager(run_index).read_attr(
                        "gradient_length_in_seconds"
                    )
                )
            except KeyError:
                value = 0.0
            if not value > 0:
                run_df = pmsm_df.filter(pl.col("run_index") == run_index)
                value = float(run_df["observed_rt"].max())
                logger.warning(
                    f"Run {run_index}: no gradient length in the result file; "
                    f"using the latest observed RT ({value:.1f} s)"
                )
            lengths[run_index] = value
        return lengths

    # ==================================================================
    # Shared pipeline steps
    # ==================================================================

    @staticmethod
    def _training_param(training_params: dict | None, name: str):
        if training_params and name in training_params:
            return training_params[name]
        return DEFAULT_TRAINING_PARAMS[name]

    @staticmethod
    def _split_group_column(level: SplitLevel) -> str | None:
        if level == "pmsm":
            return None
        if level == "precursor":
            return "precursor_index"
        if level == "peptide":
            return "peptide_index"
        raise ValueError(
            f"Unknown split level: {level!r}. "
            "Expected one of 'pmsm', 'precursor', 'peptide'."
        )

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
        group_col = TDAProcessor._split_group_column(level)
        if group_col is None:
            shuffled = pmsm_df.sample(
                fraction=1.0, with_replacement=False, shuffle=True, seed=seed
            )
            mid = shuffled.shape[0] // 2
            return shuffled.head(mid), shuffled.slice(mid)

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
    def _split_train_val(
        train_df: pl.DataFrame,
        level: SplitLevel,
        train_frac: float = 0.8,
        max_val_samples: int | None = None,
        seed: int | None = 928,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Split training rows while keeping the requested groups disjoint."""
        if not 0 < train_frac < 1:
            raise ValueError(f"train_frac must be between 0 and 1, got {train_frac}")
        if len(train_df) < 2:
            raise ValueError(
                "At least two rows are required for train/validation split"
            )

        target_val_rows = max(1, round(len(train_df) * (1 - train_frac)))
        if max_val_samples is not None:
            target_val_rows = min(target_val_rows, max_val_samples)
        target_val_rows = min(target_val_rows, len(train_df) - 1)

        group_col = TDAProcessor._split_group_column(level)
        if group_col is None:
            shuffled = train_df.sample(
                fraction=1.0,
                with_replacement=False,
                shuffle=True,
                seed=seed,
            )
            return shuffled.slice(target_val_rows), shuffled.head(target_val_rows)

        groups = (
            train_df.group_by(group_col)
            .len(name="_group_size")
            .sort(group_col)
            .sample(
                fraction=1.0,
                with_replacement=False,
                shuffle=True,
                seed=seed,
            )
        )
        if len(groups) < 2:
            raise ValueError(
                f"At least two {group_col} groups are required for a disjoint "
                "train/validation split"
            )

        cumulative_rows = groups["_group_size"].to_numpy().cumsum()[:-1]
        n_val_groups = int(np.abs(cumulative_rows - target_val_rows).argmin()) + 1
        val_group_ids = groups.head(n_val_groups).select(group_col)
        val_df = train_df.join(
            val_group_ids, on=group_col, how="inner", maintain_order="left"
        )
        fit_df = train_df.join(
            val_group_ids, on=group_col, how="anti", maintain_order="left"
        )
        return fit_df, val_df

    @staticmethod
    def _subsample_train(
        train_df: pl.DataFrame,
        level: SplitLevel = "pmsm",
        seed: int = 1221,
    ) -> pl.DataFrame:
        """Subsample rows with inverse group-frequency weights."""
        if train_df.shape[0] > TDA_MAX_TRAIN_SIZE:
            group_col = TDAProcessor._split_group_column(level)
            if group_col is None:
                train_df = train_df.sample(
                    n=TDA_MAX_TRAIN_SIZE,
                    with_replacement=False,
                    shuffle=True,
                    seed=seed,
                )
            else:
                group_counts = train_df.select(
                    pl.len().over(group_col).alias("_group_count")
                )["_group_count"].to_numpy()
                rng = np.random.default_rng(seed)
                priorities = rng.exponential(scale=group_counts)
                selected = np.argpartition(priorities, TDA_MAX_TRAIN_SIZE - 1)[
                    :TDA_MAX_TRAIN_SIZE
                ]
                train_df = train_df[selected.tolist()]
        return train_df.sort(["run_index", "pmsm_index"])

    @staticmethod
    def _build_tensor_dataset(
        df: pl.DataFrame,
        features: PmsmFeatures,
    ) -> TensorDataset:
        """Build a TensorDataset from a subset DataFrame."""
        df = df.sort(FEATURE_ROW)
        x = torch.from_numpy(features.rows(df))
        y = (~df["is_decoy"].to_torch()).float().unsqueeze(1)
        return TensorDataset(x, y)

    def _train_model(
        self,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
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
        if val_dataset is None:
            raise ValueError(
                "val_dataset is required; split the source DataFrame with "
                "_split_train_val before building tensor datasets"
            )
        trainer.train(
            model_version=model_version,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=self.output_dir,
            device=self.device,
        )
        return trainer.get_best_model().to(self.device).eval()

    @staticmethod
    def _batched_inference(
        models: list[torch.nn.Module],
        feature_arr: np.ndarray,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """``(len(models), n)`` logits; each batch is moved to the device once."""
        device = next(models[0].parameters()).device
        x_all = torch.from_numpy(feature_arr)
        logits = np.empty((len(models), len(feature_arr)), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(x_all), batch_size):
                end = min(start + batch_size, len(x_all))
                x = x_all[start:end].to(device)
                for i, model in enumerate(models):
                    logits[i, start:end] = model(x).flatten().cpu().numpy()
        return logits
