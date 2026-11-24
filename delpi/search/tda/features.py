"""Dataset-specific rescoring features appended to the PmSM embedding.

The rescoring classifier input is ``[PmSM embedding | features]``. The
features below are computed per PmSM and then z-scored with statistics of
all PmSMs of the dataset (see :class:`~delpi.search.tda.tda_processor.TDAProcessor`),
so only transforms that are not a constant rescaling matter here.
"""

import numpy as np
import polars as pl

from delpi.constants import PMSM_EMBEDDING_DIM

RESCORING_FEATURES = (
    "rt_error",  # (observed_rt - predicted_rt) / gradient length
    "precursor_mz",
    "log_median_intensity",  # log1p(MS2 median intensity); 0 if undefined
    "relative_rt",  # observed_rt / gradient length
    "sequence_length",
    "cluster_gap",  # logit minus the best logit of another precursor in the cluster
)
FEATURE_DIM = PMSM_EMBEDDING_DIM + len(RESCORING_FEATURES)
# cluster_gap of a PmSM without a competitor; gaps are clipped to +-this
CLUSTER_GAP_CAP = 10.0


def compute_rescoring_features(
    pmsm_df: pl.DataFrame, gradient_lengths: dict[int, float]
) -> np.ndarray:
    """Unscaled `RESCORING_FEATURES` for every row, as a float32 array.

    `pmsm_df` needs ``run_index``, ``observed_rt``, ``predicted_rt``,
    ``precursor_mz``, ``median_intensity``, ``sequence_length``, and for
    `cluster_gap` ``cluster``, ``precursor_index`` and ``logit`` (the
    pre-trained model's logit).
    `gradient_lengths` maps each run index to its gradient length (seconds);
    dividing by it keeps RT features comparable across runs.
    """
    gradient = pl.col("run_index").replace_strict(
        gradient_lengths, return_dtype=pl.Float64
    )
    intensity = (
        pl.col("median_intensity").fill_nan(0.0).fill_null(0.0).clip(lower_bound=0.0)
    )
    return (
        pmsm_df.select(
            ((pl.col("observed_rt") - pl.col("predicted_rt")) / gradient).alias(
                "rt_error"
            ),
            pl.col("precursor_mz"),
            intensity.log1p().alias("log_median_intensity"),
            (pl.col("observed_rt") / gradient).alias("relative_rt"),
            pl.col("sequence_length"),
        )
        .with_columns(_cluster_gap(pmsm_df))
        .cast(pl.Float32)
        .to_numpy()
    )


def _cluster_gap(pmsm_df: pl.DataFrame) -> pl.Series:
    """Logit of each PmSM minus the best logit of another precursor in its
    (run, cluster); `CLUSTER_GAP_CAP` when there is none."""
    keys = ["run_index", "cluster"]
    logit = pl.col("logit").cast(pl.Float32)
    per_precursor = pmsm_df.group_by(*keys, "precursor_index").agg(
        logit.max().alias("best")
    )
    by_best = pl.col("best").sort(descending=True)
    top = per_precursor.group_by(keys).agg(
        pl.col("precursor_index").sort_by("best", descending=True).first().alias("top"),
        by_best.first().alias("top1"),
        by_best.get(1, null_on_oob=True).alias("top2"),
    )
    competitor = (
        pl.when(pl.col("precursor_index") == pl.col("top"))
        .then(pl.col("top2"))
        .otherwise(pl.col("top1"))
    )
    return (
        pmsm_df.select(*keys, "precursor_index", logit)
        .join(top, on=keys, how="left", maintain_order="left")
        .select(
            (pl.col("logit") - competitor)
            .fill_null(CLUSTER_GAP_CAP)
            .clip(-CLUSTER_GAP_CAP, CLUSTER_GAP_CAP)
            .alias("cluster_gap")
        )
        .to_series()
    )
