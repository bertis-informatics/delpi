"""
Double-buffered batch generator for DIA search.

Uses two sets of numpy buffers so that the producer (CPU/numba) can fill
one buffer while the consumer (GPU inference) reads from the other.
This is required for safe threading-based prefetch: the original
generate_batches reuses a single buffer set, so a background thread
would overwrite data the consumer is still reading.
"""

from typing import Iterator, Tuple

import numpy as np

from delpi.lcms.data_container import DIAWindowFrameNumMap, PeakContainer
from delpi.database.numba.spec_lib_container import SpectralLibContainer
from delpi.search.dia.peak_group import (
    PeakGroupContainer,
    PeakIndexContainer,
)
from delpi.search.dia.peak_token import (
    EXP_TOKEN_DIM,
    THEO_TOKEN_DIM,
    MAX_EXP_PEAK_TOKENS,
)
from delpi.search.dia.batch_generator import (
    count_total_batches,
    iter_batch_indices,
    _make_batch_in_parallel,
)
from delpi.constants import QUANT_FRAGMENTS, RT_WINDOW_LEN

# re-export so callers can import from this module
__all__ = [
    "count_total_batches",
    "generate_batches",
]


def _allocate_buffer(batch_size: int, num_theoretical_peaks: int):
    """Allocate a complete set of numpy arrays for one batch."""
    return {
        "X_theo": np.empty(
            (batch_size, num_theoretical_peaks, THEO_TOKEN_DIM), dtype=np.float32
        ),
        "X_exp": np.empty(
            (batch_size, MAX_EXP_PEAK_TOKENS, EXP_TOKEN_DIM), dtype=np.float32
        ),
        "X_precursor_index": np.empty(batch_size, dtype=np.uint32),
        "ms1_scale_arr": np.empty(batch_size, dtype=np.float32),
        "X_indices": np.empty((batch_size, 128), dtype=np.int32),
        "X_quant": np.empty(
            (batch_size, QUANT_FRAGMENTS, RT_WINDOW_LEN), dtype=np.float32
        ),
    }


def generate_batches(
    speclib_container: SpectralLibContainer,
    ms1_peak_df: PeakContainer,
    ms2_peak_df: PeakContainer,
    peak_group_container: PeakGroupContainer,
    peak_index_container: PeakIndexContainer,
    frame_num_map: DIAWindowFrameNumMap,
    batch_size: int,
    ms1_mass_tol: float,
    ms2_mass_tol: float,
    num_buffers: int = 2,
):
    """
    Double-buffered version of generate_batches.

    Allocates ``num_buffers`` independent buffer sets and rotates through
    them so that the yielded arrays remain valid until the *next-next*
    iteration (i.e. for at least one full consumer step).

    The function signature is a superset of the original: existing callers
    that ignore ``num_buffers`` get double-buffering by default.
    """
    num_theoretical_peaks = (
        speclib_container.max_fragments + speclib_container.max_precursor_isotopes
    )

    buffers = [
        _allocate_buffer(batch_size, num_theoretical_peaks) for _ in range(num_buffers)
    ]

    frame_num_arr = peak_group_container.frame_num_arr
    batch_iter = iter_batch_indices(peak_group_container.peak_count_arr, batch_size)

    for buf_idx, (num_peaks, batch_indices) in enumerate(batch_iter):
        buf = buffers[buf_idx % num_buffers]
        cur_batch_size = batch_indices.shape[0]

        _make_batch_in_parallel(
            batch_indices,
            speclib_container,
            ms1_peak_df,
            ms2_peak_df,
            peak_group_container,
            peak_index_container,
            frame_num_map,
            buf["X_precursor_index"],
            buf["X_theo"],
            buf["X_exp"],
            buf["X_indices"],
            buf["X_quant"],
            ms1_mass_tol,
            ms2_mass_tol,
            buf["ms1_scale_arr"],
        )

        yield (
            buf["X_precursor_index"][:cur_batch_size],
            frame_num_arr[batch_indices],
            buf["X_theo"][:cur_batch_size, :, :],
            buf["X_exp"][:cur_batch_size, :num_peaks, :],
            buf["X_indices"][:cur_batch_size, :],
            buf["X_quant"][:cur_batch_size, :, :],
            buf["ms1_scale_arr"][:cur_batch_size],
        )
