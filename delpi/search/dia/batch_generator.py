from typing import Iterator, Tuple

import numba as nb
import numpy as np

from delpi.lcms.data_container import DIAWindowFrameNumMap, PeakContainer
from delpi.database.numba.spec_lib_container import SpectralLibContainer
from delpi.search.dia.peak_group import (
    PeakGroupContainer,
    PeakIndexContainer,
)
from delpi.database.numba.spec_lib_utils import get_theoretical_peaks
from delpi.search.dia.peak_token import (
    get_x_exp,
    get_x_theo,
    EXP_TOKEN_DIM,
    THEO_TOKEN_DIM,
    MAX_EXP_PEAK_TOKENS,
)


def count_total_batches(peak_counts_arr: np.ndarray, batch_size: int = 512) -> int:
    unique, counts = np.unique(peak_counts_arr, return_counts=True)
    total = 0
    for c in counts:
        total += (c + batch_size - 1) // batch_size  # == ceil(c / batch_size)
    return total


def iter_batch_indices(
    peak_counts_arr: np.ndarray, batch_size: int = 512
) -> Iterator[Tuple[int, np.ndarray]]:

    sorted_idx = np.argsort(peak_counts_arr)
    sorted_peak_counts = peak_counts_arr[sorted_idx]
    n = len(sorted_peak_counts)

    start = 0
    while start < n:
        current_value = sorted_peak_counts[start]
        end = start + 1
        while end < n and sorted_peak_counts[end] == current_value:
            end += 1

        group_indices = sorted_idx[start:end]
        for i in range(0, end - start, batch_size):
            yield current_value, group_indices[i : i + batch_size]
        start = end


@nb.njit(parallel=True, cache=True)
def _make_batch_in_parallel(
    batch_indices: np.ndarray,
    speclib_container: SpectralLibContainer,
    ms1_peak_df: PeakContainer,
    ms2_peak_df: PeakContainer,
    peak_group_container: PeakGroupContainer,
    peak_index_container: PeakIndexContainer,
    frame_num_map: DIAWindowFrameNumMap,
    X_precursor_index: np.ndarray,
    X_theo: np.ndarray,
    X_exp: np.ndarray,
    X_indices: np.ndarray,
    X_quant: np.ndarray,
    ms1_mass_tol: float,
    ms2_mass_tol: float,
    ms1_scale_arr: np.ndarray,
):
    frame_num_arr = peak_group_container.frame_num_arr
    precursor_index0_arr = peak_group_container.precursor_index0_arr
    min_precursor_index = speclib_container.min_precursor_index

    cur_batch_size = batch_indices.shape[0]
    X_indices[:] = -1
    X_quant[:] = -1
    ms1_scale_arr[:] = -1.0

    # for i, k in enumerate(batch_indices):
    for i in nb.prange(cur_batch_size):
        k = batch_indices[i]
        frame_num = frame_num_arr[k]
        precursor_index0 = precursor_index0_arr[k]

        theo_peaks = get_theoretical_peaks(speclib_container, precursor_index0)
        _ = get_x_theo(theo_peaks, X_theo[i])
        _, ms1_scale = get_x_exp(
            precursor_index0=precursor_index0,
            frame_num=frame_num,
            theo_peaks=theo_peaks,
            ms1_peak_df=ms1_peak_df,
            ms2_peak_df=ms2_peak_df,
            frame_num_map=frame_num_map,
            rt_window_radius=4,
            peak_index_container=peak_index_container,
            ms1_mass_tol=ms1_mass_tol,
            ms2_mass_tol=ms2_mass_tol,
            x_exp=X_exp[i],
            x_ind=X_indices[i],
            x_quant=X_quant[i],
        )
        X_precursor_index[i] = precursor_index0 + min_precursor_index
        ms1_scale_arr[i] = ms1_scale


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
):

    num_theoretical_peaks = (
        speclib_container.max_fragments + speclib_container.max_precursor_isotopes
    )

    X_theo = np.empty(
        (batch_size, num_theoretical_peaks, THEO_TOKEN_DIM),
        dtype=np.float32,
    )
    X_exp = np.empty(
        (batch_size, MAX_EXP_PEAK_TOKENS, EXP_TOKEN_DIM),
        dtype=np.float32,
    )
    X_precursor_index = np.empty(batch_size, dtype=np.uint32)
    ms1_scale_arr = np.empty(batch_size, dtype=np.float32)

    # Peak indices for matched fragment ions
    X_indices = np.empty((batch_size, 128), dtype=np.int32)

    # Fragment peak intensities for quantification
    X_quant = np.empty((batch_size, 128, 3), dtype=np.float32)

    frame_num_arr = peak_group_container.frame_num_arr
    # precursor_index0_arr = peak_group_container.precursor_index0_arr
    # min_precursor_index = frag_db.min_precursor_index
    batch_iter = iter_batch_indices(peak_group_container.peak_count_arr, batch_size)

    for num_peaks, batch_indices in batch_iter:
        cur_batch_size = batch_indices.shape[0]
        _make_batch_in_parallel(
            batch_indices,
            speclib_container,
            ms1_peak_df,
            ms2_peak_df,
            peak_group_container,
            peak_index_container,
            frame_num_map,
            X_precursor_index,
            X_theo,
            X_exp,
            X_indices,
            X_quant,
            ms1_mass_tol,
            ms2_mass_tol,
            ms1_scale_arr,
        )

        yield (
            X_precursor_index[:cur_batch_size],
            frame_num_arr[batch_indices],
            X_theo[:cur_batch_size, :, :],
            X_exp[:cur_batch_size, :num_peaks, :],
            X_indices[:cur_batch_size, :],
            X_quant[:cur_batch_size, :, :],
            ms1_scale_arr[:cur_batch_size],
        )
