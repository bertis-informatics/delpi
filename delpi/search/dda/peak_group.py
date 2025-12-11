from typing import Tuple, NamedTuple

import numpy as np
import numba as nb

from delpi.lcms.data_container import PeakContainer, MetaContainer
from delpi.database.numba.spec_lib_container import SpectralLibContainer
from delpi.search.dia.peak_group import PeakIndexContainer
from delpi.database.numba.spec_lib_utils import get_theoretical_peaks
from delpi.utils.peak import find_peak_index

MATCHED_PEAKS_CUTOFF_LB = 3
MATCHED_PEAKS_CUTOFF_UB = 6


class DdaPeakGroupContainer(NamedTuple):
    min_precursor_index: int
    precursor_index0_arr: np.ndarray
    frame_num_arr: np.ndarray
    peak_count_arr: np.ndarray
    frame_num_key_arr: np.ndarray
    time_index_val_arr: np.ndarray


@nb.njit(inline="always", nogil=True, fastmath=True, cache=True)
def get_matched_peak_cutoff(pept_len: int):
    return max(
        MATCHED_PEAKS_CUTOFF_LB, min(MATCHED_PEAKS_CUTOFF_UB, np.ceil(pept_len / 3))
    )


@nb.njit(nogil=True, fastmath=True, cache=True)
def _assign_spectra_to_bins(rt_window_center, rt_window_half, rt_bin_count, rt_array):
    """
    For each of `rt_bin_count` bins, choose the spectrum with the highest score
    whose RT falls inside [centre ± rt_window_half].
    Returns uint32 indices; 0xFFFFFFFF means “no spectrum in that bin”.
    """
    n_spec = rt_array.shape[0]
    assigned = np.full(rt_bin_count, -1, dtype=np.int32)

    # Edge-case: no spectra or bins
    if n_spec == 0 or rt_bin_count <= 0:
        return assigned

    # Bin spacing so the last bin's centre hits max(rt_array)
    # max_rt = rt_array[n_spec - 1]
    bin_size = (rt_window_half * 2) / (rt_bin_count - 1)
    bin_half = bin_size * 0.5

    best_rt_diff = np.full(rt_bin_count, 1e9, dtype=np.float32)
    for b in range(rt_bin_count):
        centre = rt_window_center - rt_window_half + b * bin_size
        left = np.searchsorted(rt_array, centre - bin_half, side="left")
        right = np.searchsorted(rt_array, centre + bin_half, side="right")

        for i in range(left, right):
            rt_diff = np.abs(centre - rt_array[i])
            if rt_diff < best_rt_diff[b]:
                assigned[b] = i  # cast to uint32
                best_rt_diff[b] = rt_diff

    return assigned


@nb.njit(cache=True)
def lookup_time_index(
    frame_num_key_arr: np.ndarray,
    time_index_val_arr: np.ndarray,
    frame_num: int,
):
    first_key = frame_num_key_arr[0]
    if frame_num < first_key or first_key == -1:
        return -1

    for i in range(frame_num_key_arr.shape[0]):
        k = frame_num_key_arr[i]
        if k == frame_num:
            return time_index_val_arr[i]
        if k < 0 or k > frame_num:
            break

    return -1


@nb.njit(nogil=True, fastmath=True, cache=True)
def _generate_peak_group_arrays(
    peak_group_ii: np.ndarray,
    ms1_meta_df: MetaContainer,
    ms2_meta_df: MetaContainer,
    ms2_frame_indexes: np.ndarray,
    rt_window_half: float,
    rt_bin_count: int = 9,
):
    peak_group_count = peak_group_ii.shape[0]
    frame_num_arr = ms2_meta_df.frame_num_arr[ms2_frame_indexes[peak_group_ii]]
    ms2_rt_array = ms2_meta_df.rt_arr[ms2_frame_indexes]
    ms1_rt_array = ms1_meta_df.rt_arr

    frame_num_key_arr = np.full((peak_group_count, 2, rt_bin_count), -1, dtype=np.int32)
    time_index_val_arr = np.full((peak_group_count, 2, rt_bin_count), -1, dtype=np.int8)

    for j, ii in enumerate(peak_group_ii):
        peak_group_rt = ms2_rt_array[ii]
        ms2_assigned = _assign_spectra_to_bins(
            peak_group_rt, rt_window_half, rt_bin_count, ms2_rt_array
        )
        ms1_assigned = _assign_spectra_to_bins(
            peak_group_rt, rt_window_half, rt_bin_count, ms1_rt_array
        )

        ms1_fr_cnt, ms2_fr_cnt = 0, 0
        for time_idx, (ms1_idx, ms2_idx) in enumerate(zip(ms1_assigned, ms2_assigned)):
            if ms1_idx >= 0:
                fn = ms1_meta_df.frame_num_arr[ms1_idx]
                frame_num_key_arr[j, 0, ms1_fr_cnt] = fn
                time_index_val_arr[j, 0, ms1_fr_cnt] = time_idx
                ms1_fr_cnt += 1

            if ms2_idx >= 0:
                fn = ms2_meta_df.frame_num_arr[ms2_frame_indexes[ms2_idx]]
                frame_num_key_arr[j, 1, ms2_fr_cnt] = fn
                time_index_val_arr[j, 1, ms2_fr_cnt] = time_idx
                ms2_fr_cnt += 1

    return (
        frame_num_arr,
        frame_num_key_arr,
        time_index_val_arr,
    )


@nb.njit(nogil=True, fastmath=True, cache=True)
def iter_peaks(
    meta_df: MetaContainer,
    peak_df: PeakContainer,
    st_indices: np.ndarray,
    ed_indices: np.ndarray,
    frame_num_key_arr: np.ndarray,
    time_index_val_arr: np.ndarray,
):
    for st, ed in zip(st_indices, ed_indices):
        for i in nb.prange(st, ed):
            fn = peak_df.frame_num_arr[i]
            if lookup_time_index(frame_num_key_arr, time_index_val_arr, fn) >= 0:
                yield i


@nb.njit(nogil=True, fastmath=True, cache=True)
def count_peaks(
    frame_num_key_arr: np.ndarray,
    time_index_val_arr: np.ndarray,
    ms1_meta_df: MetaContainer,
    ms2_meta_df: MetaContainer,
    ms1_peak_df: PeakContainer,
    ms2_peak_df: PeakContainer,
    ms1_pc_st_indices: np.ndarray,
    ms1_pc_ed_indices: np.ndarray,
    ms2_pc_st_indices: np.ndarray,
    ms2_pc_ed_indices: np.ndarray,
    ms2_fg_st_indices: np.ndarray,
    ms2_fg_ed_indices: np.ndarray,
):

    n_pc = 0
    n_fg = 0
    for _ in iter_peaks(
        ms1_meta_df,
        ms1_peak_df,
        ms1_pc_st_indices,
        ms1_pc_ed_indices,
        frame_num_key_arr[0],
        time_index_val_arr[0],
    ):
        n_pc += 1

    for _ in iter_peaks(
        ms2_meta_df,
        ms2_peak_df,
        ms2_pc_st_indices,
        ms2_pc_ed_indices,
        frame_num_key_arr[1],
        time_index_val_arr[1],
    ):
        n_pc += 1

    for _ in iter_peaks(
        ms2_meta_df,
        ms2_peak_df,
        ms2_fg_st_indices,
        ms2_fg_ed_indices,
        frame_num_key_arr[1],
        time_index_val_arr[1],
    ):
        n_fg += 1

    return n_pc, n_fg


@nb.njit(nogil=True, fastmath=True, cache=True)
def _find_ms2_spectra(
    ms2_meta_df: MetaContainer,
    ms2_peak_df: PeakContainer,
    st_indices: np.ndarray,
    ed_indices: np.ndarray,
    ms2_frame_indexes: np.ndarray,
    min_peak_count: int,
    topk: int = 10,
) -> np.ndarray:
    """Find MS2 spectra with at least `min_peak_count` matched peaks

    Args:
        ms2_meta_df (MetaContainer): _description_
        ms2_peak_df (PeakContainer): _description_
        st_indices (np.ndarray): _description_
        ed_indices (np.ndarray): _description_
        ms2_frame_indexes (np.ndarray): _description_
        min_peak_count (int): _description_

    Returns:
        np.ndarray: _description_
    """

    ms2_frame_num_to_index = ms2_meta_df.frame_num_to_index_arr
    min_idx, max_idx = ms2_frame_indexes[0], ms2_frame_indexes[-1]
    mono_frag_pk_count = np.full(max_idx - min_idx + 1, -1, dtype=np.int8)
    for i in ms2_frame_indexes:
        mono_frag_pk_count[i - min_idx] = 0

    # ms2_frame_num_to_index = dda_run.ms2_map.frame_num_to_index
    for st, ed in zip(st_indices, ed_indices):
        for pk_i in range(st, ed):
            ms2_frame_idx = ms2_frame_num_to_index[ms2_peak_df.frame_num_arr[pk_i]]
            if (
                (min_idx <= ms2_frame_idx)
                and (ms2_frame_idx <= max_idx)
                and (mono_frag_pk_count[ms2_frame_idx - min_idx] >= 0)
            ):
                mono_frag_pk_count[ms2_frame_idx - min_idx] += 1

    ms2_ii = np.empty(ms2_frame_indexes.shape[0], dtype=np.uint32)
    ms2_score = np.empty(ms2_frame_indexes.shape[0], dtype=np.int8)

    ms2_ii_cnt = 0
    for ii, idx in enumerate(ms2_frame_indexes):
        s = mono_frag_pk_count[idx - min_idx]
        if s >= min_peak_count:
            ms2_ii[ms2_ii_cnt] = ii
            ms2_score[ms2_ii_cnt] = s
            ms2_ii_cnt += 1

    ms2_ii = ms2_ii[:ms2_ii_cnt]
    ms2_score = ms2_score[:ms2_ii_cnt]

    if ms2_ii_cnt > topk:
        ind = np.argsort(ms2_score)[::-1]
        ms2_ii = ms2_ii[ind[:topk]]

    return ms2_ii


@nb.njit(parallel=True, fastmath=True, cache=True)
def find_peak_groups(
    speclib_container: SpectralLibContainer,
    ms1_meta_df: MetaContainer,
    ms2_meta_df: MetaContainer,
    ms1_peak_df: PeakContainer,
    ms2_peak_df: PeakContainer,
    rt_window_half: float,
    isolation_lower_tol_in_da: float = 3.5,
    isolation_upper_tol_in_da: float = 1.25,
    ms1_mass_tol: float = 10,
    ms2_mass_tol: float = 10,
    rt_window_radius: int = 4,
    min_peak_count: int = MATCHED_PEAKS_CUTOFF_LB,
    topk: int = 10,
) -> Tuple[DdaPeakGroupContainer, PeakIndexContainer]:

    rt_bin_count = rt_window_radius * 2 + 1

    num_fragments = speclib_container.max_fragments
    num_precursors = speclib_container.precursor_mz_arr.shape[0]
    min_precursor_index = speclib_container.min_precursor_index
    num_precursor_peaks = speclib_container.max_precursor_isotopes
    num_fragment_peaks = (
        speclib_container.max_fragments * speclib_container.max_fragment_isotopes
    )

    precursor_ms1_indices = np.empty(
        (num_precursors, 2, num_precursor_peaks), dtype=np.uint32
    )
    precursor_ms2_indices = np.empty(
        (num_precursors, 2, num_precursor_peaks), dtype=np.uint32
    )
    fragment_ms2_indices = np.empty(
        (num_precursors, 2, num_fragment_peaks),
        dtype=np.uint32,
    )

    #### pre-allocated arrays for peak groups
    frame_num_results = [np.empty(0, dtype=np.uint32)] * num_precursors
    peak_count_results = [np.empty(0, dtype=np.uint16)] * num_precursors

    frame_num_key_results = [np.empty((0, 0, 0), dtype=np.int32)] * num_precursors
    time_index_val_results = [np.empty((0, 0, 0), dtype=np.int8)] * num_precursors

    for precursor_index0 in nb.prange(num_precursors):
        # rt_lb, rt_ub = frag_db.get_rt_range(precursor_index0)
        theo_peaks = get_theoretical_peaks(speclib_container, precursor_index0)

        precursor_charge = theo_peaks.precursor_charge
        precursor_mz = theo_peaks.precursor_mz_arr[0]
        min_matched = max(
            min_peak_count, get_matched_peak_cutoff(theo_peaks.sequence_length)
        )

        ## 1. Find candidate MS2 spectra
        ms2_frame_indexes = np.where(
            (
                ms2_meta_df.isolation_min_mz_arr
                - isolation_lower_tol_in_da / precursor_charge
                < precursor_mz
            )
            & (
                precursor_mz
                < ms2_meta_df.isolation_max_mz_arr
                + isolation_upper_tol_in_da / precursor_charge
            )
        )[0].astype(np.uint32)

        if ms2_frame_indexes.shape[0] < 1:
            continue

        ## 2. Peak matching
        ms2_st, ms2_ed = find_peak_index(
            ms2_peak_df.mz_arr, theo_peaks.fragment_mz_arr.flatten(), ms2_mass_tol
        )
        peak_group_ii = _find_ms2_spectra(
            ms2_meta_df,
            ms2_peak_df,
            ms2_st[:num_fragments],
            ms2_ed[:num_fragments],
            ms2_frame_indexes,
            min_matched,
            topk,
        )

        if peak_group_ii.shape[0] < 1:
            continue

        # precursor ion peaks in MS1
        pc_ms1_st, pc_ms1_ed = find_peak_index(
            ms1_peak_df.mz_arr, theo_peaks.precursor_mz_arr, ms1_mass_tol
        )
        # precursor ion peaks in MS2
        pc_ms2_st, pc_ms2_ed = find_peak_index(
            ms2_peak_df.mz_arr, theo_peaks.precursor_mz_arr, ms2_mass_tol
        )
        # store matching results
        fragment_ms2_indices[precursor_index0, 0] = ms2_st
        fragment_ms2_indices[precursor_index0, 1] = ms2_ed
        precursor_ms1_indices[precursor_index0, 0] = pc_ms1_st
        precursor_ms1_indices[precursor_index0, 1] = pc_ms1_ed
        precursor_ms2_indices[precursor_index0, 0] = pc_ms2_st
        precursor_ms2_indices[precursor_index0, 1] = pc_ms2_ed

        ## 3. Generate RT groups
        (
            frame_num_arr,
            frame_num_key_arr,
            time_index_val_arr,
        ) = _generate_peak_group_arrays(
            peak_group_ii,
            ms1_meta_df,
            ms2_meta_df,
            ms2_frame_indexes,
            rt_window_half,
            rt_bin_count=rt_bin_count,
        )

        n_groups = frame_num_arr.shape[0]
        peak_count_arr = np.zeros(n_groups, dtype=np.uint16)
        for k in range(n_groups):
            precursor_pk_cnt, fragment_pk_cnt = count_peaks(
                frame_num_key_arr[k],
                time_index_val_arr[k],
                ms1_meta_df,
                ms2_meta_df,
                ms1_peak_df,
                ms2_peak_df,
                ms1_pc_st_indices=pc_ms1_st,
                ms1_pc_ed_indices=pc_ms1_ed,
                ms2_pc_st_indices=pc_ms2_st,
                ms2_pc_ed_indices=pc_ms2_ed,
                ms2_fg_st_indices=ms2_st,
                ms2_fg_ed_indices=ms2_ed,
            )
            peak_count_arr[k] = precursor_pk_cnt + fragment_pk_cnt

        frame_num_results[precursor_index0] = frame_num_arr
        peak_count_results[precursor_index0] = peak_count_arr
        frame_num_key_results[precursor_index0] = frame_num_key_arr
        time_index_val_results[precursor_index0] = time_index_val_arr

    num_peak_groups = np.cumsum(
        np.array([arr.shape[0] for arr in peak_count_results], dtype=np.uint32)
    )

    n = num_peak_groups[-1]
    precursor_index0_arr = np.empty(n, dtype=np.uint32)
    peak_count_arr = np.empty(n, dtype=np.uint16)
    frame_num_arr = np.empty(n, dtype=np.uint32)
    frame_num_key_arr = np.empty((n, 2, rt_bin_count), dtype=np.int32)
    time_index_val_arr = np.empty((n, 2, rt_bin_count), dtype=np.int8)

    for i in nb.prange(num_precursors):
        st = 0 if i == 0 else num_peak_groups[i - 1]
        ed = num_peak_groups[i]
        if ed > st:
            precursor_index0_arr[st:ed] = i
            peak_count_arr[st:ed] = peak_count_results[i]
            frame_num_arr[st:ed] = frame_num_results[i]
            frame_num_key_arr[st:ed] = frame_num_key_results[i]
            time_index_val_arr[st:ed] = time_index_val_results[i]

    frame_num_results.clear()
    peak_count_results.clear()
    frame_num_key_results.clear()
    time_index_val_results.clear()

    peak_group_container = DdaPeakGroupContainer(
        min_precursor_index,
        precursor_index0_arr,
        frame_num_arr,
        peak_count_arr,
        frame_num_key_arr,
        time_index_val_arr,
    )

    peak_index_container = PeakIndexContainer(
        precursor_ms1_indices,
        precursor_ms2_indices,
        fragment_ms2_indices,
    )

    return peak_group_container, peak_index_container
