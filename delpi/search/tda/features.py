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
)
FEATURE_DIM = PMSM_EMBEDDING_DIM + len(RESCORING_FEATURES)


def compute_rescoring_features(
    pmsm_df: pl.DataFrame, gradient_lengths: dict[int, float]
) -> np.ndarray:
    """Unscaled `RESCORING_FEATURES` for every row, as a float32 array.

    `pmsm_df` needs ``run_index``, ``observed_rt``, ``predicted_rt``,
    ``precursor_mz``, ``median_intensity`` and ``sequence_length``.
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
        .cast(pl.Float32)
        .to_numpy()
    )
