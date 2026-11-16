"""Shared on-disk spectral library schema and precursor row indexing."""

import numpy as np
import polars as pl

SPECLIB_SCHEMA = {
    "precursor_index": pl.UInt32,
    "cleavage_index": pl.UInt8,
    "is_prefix": pl.Boolean,
    "charge": pl.UInt8,
    "is_modloss": pl.Boolean,
    "mz": pl.Float32,
    "predicted_intensity": pl.Float32,
    "rank": pl.UInt8,  # Descending predicted intensity (1 is strongest).
}


def fragment_offsets(counts: np.ndarray, start: int = 0) -> np.ndarray:
    """Build offsets without silently overflowing the on-disk UInt32 type."""
    stops = np.cumsum(counts, dtype=np.uint64) + start
    if stops.size and stops[-1] > np.iinfo(np.uint32).max:
        raise ValueError("Spectral library row offsets exceed the UInt32 limit.")
    return (stops - counts).astype(np.uint32)


def with_speclib_index(precursor_df: pl.DataFrame, counts: np.ndarray) -> pl.DataFrame:
    if len(counts) != precursor_df.height:
        raise ValueError("Fragment counts must match the precursor rows.")
    return precursor_df.with_columns(
        pl.Series("speclib_offset", fragment_offsets(counts), dtype=pl.UInt32),
        pl.Series("speclib_rows", counts, dtype=pl.UInt8),
    )
