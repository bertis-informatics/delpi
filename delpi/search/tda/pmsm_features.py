"""Rescoring classifier inputs: [PmSM embedding | z-scored rescoring features].

Rows are addressed by ``FEATURE_ROW``, the row number of the PmSM table
sorted by (run_index, pmsm_index). In memory, all inputs are read once (one
HDF read per run); otherwise each request reads its rows run by run.
"""

from collections.abc import Iterator

import numpy as np
import polars as pl

from delpi.constants import PMSM_EMBEDDING_DIM
from delpi.search.result_aggregator import ResultsAggregator
from delpi.search.tda.features import FEATURE_DIM

FEATURE_ROW = "_feature_row"


class PmsmFeatures:
    """Classifier inputs of the PmSMs in `pmsm_df` (sorted by `FEATURE_ROW`).
    `rescoring_features` row i belongs to `FEATURE_ROW` i."""

    def __init__(
        self,
        result_aggregator: ResultsAggregator,
        group_key: str,
        pmsm_df: pl.DataFrame,
        rescoring_features: np.ndarray,
        in_memory: bool,
    ):
        self._aggregator = result_aggregator
        self._group_key = group_key
        self._pmsm_df = pmsm_df
        self._rescoring_features = rescoring_features
        self._matrix = self._read(pmsm_df) if in_memory else None

    @property
    def in_memory(self) -> bool:
        return self._matrix is not None

    def rows(self, df: pl.DataFrame) -> np.ndarray:
        """Inputs of `df`'s rows (`df` sorted by `FEATURE_ROW`)."""
        if self._matrix is not None:
            return self._matrix[df[FEATURE_ROW].to_numpy()]
        return self._read(df)

    def run_blocks(self) -> Iterator[tuple[int, int, np.ndarray]]:
        """``(start, end, inputs)`` of each run's rows ``start:end``."""
        for _, start, end, run_df in _run_blocks(self._pmsm_df):
            if self._matrix is not None:
                yield start, end, self._matrix[start:end]
            else:
                yield start, end, self._read(run_df)

    def _read(self, df: pl.DataFrame) -> np.ndarray:
        """Read `df`'s rows from the HDF files, one read per run."""
        inputs = np.empty((df.height, FEATURE_DIM), dtype=np.float32)
        for run_index, start, end, run_df in _run_blocks(df):
            self._aggregator.get_result_manager(run_index).load_features(
                self._group_key,
                feature_dim=FEATURE_DIM,
                out=inputs[start:end],
                row_indices=run_df["pmsm_index"].to_numpy(),
            )
        inputs[:, PMSM_EMBEDDING_DIM:] = self._rescoring_features[
            df[FEATURE_ROW].to_numpy()
        ]
        return inputs


def _run_blocks(df: pl.DataFrame) -> Iterator[tuple[int, int, int, pl.DataFrame]]:
    """``(run_index, start, end, run_df)`` of each run's rows in `df`."""
    start = 0
    for (run_index,), run_df in df.group_by("run_index", maintain_order=True):
        end = start + run_df.height
        yield run_index, start, end, run_df
        start = end
