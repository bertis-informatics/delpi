from typing import Tuple
from typing import NamedTuple

import numpy as np
import numba as nb

from delpi.lcms.data_container import DIAWindowFrameNumMap, PeakContainer

from delpi.database.numba.spec_lib_container import SpectralLibContainer
from delpi.database.numba.spec_lib_utils import (
    get_frame_index_range,
    get_theoretical_peaks,
)
from delpi.utils.signal import find_local_maxima, cluster_peaks_with_weights
from delpi.utils.peak import find_peak_index
from delpi.utils.numeric import corrcoef

QUANT_FRAGMENTS = 12


class PeakIndexContainer(NamedTuple):
    precursor_ms1_indices: np.ndarray
    precursor_ms2_indices: np.ndarray
    fragment_ms2_indices: np.ndarray


class PeakGroupContainer(NamedTuple):
    min_precursor_index: int
    precursor_index0_arr: np.ndarray
    frame_num_arr: np.ndarray
    peak_count_arr: np.ndarray


@nb.njit(nogil=True, fastmath=True, cache=True)
def _count_peaks(
    peak_df: PeakContainer,
    frame_num_map: DIAWindowFrameNumMap,
    st_indices: np.ndarray,
    ed_indices: np.ndarray,
    min_frame_index: int,
    max_frame_index: int,
    out_arr: np.ndarray,
):

    for st, ed in zip(st_indices, ed_indices):
        for fn in peak_df.frame_num_arr[st:ed]:
            fi = frame_num_map.frame_num_to_index_arr[fn]
            if (fi >= min_frame_index) and (fi <= max_frame_index):
                out_arr[fi - min_frame_index] += 1

    return out_arr


@nb.njit(nogil=True, fastmath=True, cache=True)
def count_peaks(
    ms1_peak_df: PeakContainer,
    ms2_peak_df: PeakContainer,
    frame_num_map: DIAWindowFrameNumMap,
    ms1_pc_st_indices: np.ndarray,
    ms1_pc_ed_indices: np.ndarray,
    ms2_pc_st_indices: np.ndarray,
    ms2_pc_ed_indices: np.ndarray,
    ms2_fg_st_indices: np.ndarray,
    ms2_fg_ed_indices: np.ndarray,
    min_frame_index: int,
    max_frame_index: int,
):
    xic_len = max_frame_index - min_frame_index + 1
    all_peak_count_arr = np.zeros(xic_len, dtype=np.uint16)

    _ = _count_peaks(
        ms2_peak_df,
        frame_num_map,
        ms2_fg_st_indices,
        ms2_fg_ed_indices,
        min_frame_index,
        max_frame_index,
        all_peak_count_arr,
    )

    # count all precursor peaks in MS1 & MS2
    _ = _count_peaks(
        ms1_peak_df,
        frame_num_map,
        ms1_pc_st_indices,
        ms1_pc_ed_indices,
        min_frame_index,
        max_frame_index,
        all_peak_count_arr,
    )
    _ = _count_peaks(
        ms2_peak_df,
        frame_num_map,
        ms2_pc_st_indices,
        ms2_pc_ed_indices,
        min_frame_index,
        max_frame_index,
        all_peak_count_arr,
    )
    return all_peak_count_arr


@nb.njit(nogil=True, fastmath=True, cache=True)
def make_xic_array(
    peak_df: PeakContainer,
    frame_num_map: DIAWindowFrameNumMap,
    st_indices: int,
    ed_indices: int,
    min_frame_index: int,
    max_frame_index: int,
    num_fragments: int,
) -> np.ndarray:
    """_summary_

    Args:
        max_xics (int): _description_
        min_frame_index (int): _description_
        max_frame_index (int): _description_
        theo_peak_arr (np.ndarray): _description_
        exp_peak_arr (np.ndarray): _description_

    Returns:
        np.ndarray: shape of (2, fragments, time points)
    """

    xic_len = max_frame_index - min_frame_index + 1
    xic_arr = np.zeros((num_fragments, xic_len), dtype=np.float32)
    # rank_arr = theo_peaks.fragment_rank_arr

    for i in nb.prange(num_fragments):
        st, ed = st_indices[i], ed_indices[i]
        for j in range(st, ed):
            fn = peak_df.frame_num_arr[j]
            fi = frame_num_map.frame_num_to_index_arr[fn]
            if (min_frame_index <= fi) and (fi <= max_frame_index):
                ab = peak_df.ab_arr[j]
                time_idx = fi - min_frame_index
                xic_arr[i, time_idx] = max(ab, xic_arr[i, time_idx])

    return xic_arr


@nb.njit(nogil=True, fastmath=True, cache=True)
def count_xic_peaks(dense_arr: np.ndarray):

    num_xics = dense_arr.shape[0]
    xic_peak_count_arr = np.zeros(dense_arr.shape[1], dtype=np.uint16)

    detected_peaks = [np.empty(0, dtype=np.uint32)] * num_xics
    for i in nb.prange(num_xics):
        xic = dense_arr[i]
        if np.any(xic > 0):
            peaks, left_edges, right_edges = find_local_maxima(xic)
            detected_peaks[i] = peaks

    for peaks in detected_peaks:
        for j in peaks:
            xic_peak_count_arr[j] += 1

    return xic_peak_count_arr


@nb.njit(nogil=True, fastmath=True, cache=True)
def clip_peak_group(peak_groups: np.ndarray, peak_lb: int, peak_ub: int):

    assert np.all(peak_groups[:-1] <= peak_groups[1:]), "peak_groups should be ordered"
    n = peak_groups.shape[0]

    result = np.empty_like(peak_groups)
    index = 0

    i = 0
    while i < n and peak_groups[i] <= peak_lb:
        i += 1

    # val < lower_bound
    if i > 0:
        result[index] = peak_lb
        index += 1

    # copy for lb < values <= ub
    while i < n and peak_groups[i] < peak_ub:
        result[index] = peak_groups[i]
        index += 1
        i += 1

    # val > upper_bound
    if i < n:
        result[index] = peak_ub
        index += 1

    return result[:index]


@nb.njit(nogil=True, fastmath=True, cache=True)
def get_avg_xic_corrs(
    xic_arr: np.ndarray,
    peak_group_arr: np.ndarray,
    rt_window_radius: int,
):
    corr_score = np.empty(peak_group_arr.shape[0], dtype=np.float32)

    for i in nb.prange(peak_group_arr.shape[0]):
        peak = peak_group_arr[i]
        st = peak - rt_window_radius
        ed = peak + rt_window_radius + 1

        xic_arr_ = xic_arr[:, st:ed]
        mask = np.sum(xic_arr_, axis=1) > 0
        n_xics = np.sum(mask)
        if n_xics < 2:
            corr_score[i] = 0
        else:
            corr_arr = corrcoef(xic_arr_[mask])
            n_corrs = n_xics * (n_xics - 1) // 2
            corr_power_3_sum = 0.0
            for j in range(n_xics):
                for k in range(j + 1, n_xics):
                    corr_power_3_sum += corr_arr[j, k] ** 3
            corr_score[i] = corr_power_3_sum / n_corrs

    return corr_score


@nb.njit(nogil=True, fastmath=True, cache=True)
def sum_neighborhood(all_peak_count_arr, rt_window_radius, peak_group_arr):
    n = peak_group_arr.shape[0]
    sum_peak_count_arr = np.empty(n, dtype=all_peak_count_arr.dtype)
    for i in nb.prange(n):
        pk_group = peak_group_arr[i]
        st = pk_group - rt_window_radius
        ed = pk_group + rt_window_radius + 1
        sum_peak_count_arr[i] = all_peak_count_arr[st:ed].sum()
    return sum_peak_count_arr


@nb.njit(parallel=True, fastmath=True, cache=True)
def find_peak_groups(
    speclib_container: SpectralLibContainer,
    ms1_peak_df: PeakContainer,
    ms2_peak_df: PeakContainer,
    frame_num_map: DIAWindowFrameNumMap,
    ms1_mass_tol: float = 10,
    ms2_mass_tol: float = 10,
    rt_window_radius: int = 4,
    min_xic_peak_count: int = 3,
    min_peak_count: int = 16,
    topk: int = 10,
) -> Tuple[PeakGroupContainer, PeakIndexContainer]:

    num_fragments = speclib_container.max_fragments
    num_precursors = speclib_container.precursor_mz_arr.shape[0]
    min_precursor_index = speclib_container.min_precursor_index
    peak_group_results = [np.empty(0, dtype=np.uint32)] * num_precursors
    peak_count_results = [np.empty(0, dtype=np.uint16)] * num_precursors
    num_peak_groups = np.zeros(num_precursors, dtype=np.uint32)
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
        (num_precursors, 2, num_fragment_peaks), dtype=np.uint32
    )

    for precursor_index0 in nb.prange(num_precursors):

        min_frame_index, max_frame_index = get_frame_index_range(
            speclib_container, frame_num_map.ms2_rt_arr, precursor_index0
        )

        xic_len = max_frame_index - min_frame_index + 1
        peak_lb = rt_window_radius  # inclusive
        peak_ub = xic_len - rt_window_radius - 1  # inclusive

        # searching for fragment ion peaks in MS2
        theo_peaks = get_theoretical_peaks(speclib_container, precursor_index0)

        fragment_mz_arr = theo_peaks.fragment_mz_arr
        ms2_st, ms2_ed = find_peak_index(
            ms2_peak_df.mz_arr, fragment_mz_arr.flatten(), ms2_mass_tol
        )
        # searching for precursor ion peaks in MS1
        mz_arr = theo_peaks.precursor_mz_arr
        pc_ms1_st, pc_ms1_ed = find_peak_index(ms1_peak_df.mz_arr, mz_arr, ms1_mass_tol)
        pc_ms2_st, pc_ms2_ed = find_peak_index(ms2_peak_df.mz_arr, mz_arr, ms2_mass_tol)

        # store matching results
        fragment_ms2_indices[precursor_index0, 0] = ms2_st
        fragment_ms2_indices[precursor_index0, 1] = ms2_ed
        precursor_ms1_indices[precursor_index0, 0] = pc_ms1_st
        precursor_ms1_indices[precursor_index0, 1] = pc_ms1_ed
        precursor_ms2_indices[precursor_index0, 0] = pc_ms2_st
        precursor_ms2_indices[precursor_index0, 1] = pc_ms2_ed

        all_peak_count_arr = count_peaks(
            ms1_peak_df,
            ms2_peak_df,
            frame_num_map,
            ms1_pc_st_indices=pc_ms1_st,
            ms1_pc_ed_indices=pc_ms1_ed,
            ms2_pc_st_indices=pc_ms2_st,
            ms2_pc_ed_indices=pc_ms2_ed,
            ms2_fg_st_indices=ms2_st,
            ms2_fg_ed_indices=ms2_ed,
            min_frame_index=min_frame_index,
            max_frame_index=max_frame_index,
        )

        ## create fragment XIC array with fragment mono-isotopic peaks
        xic_arr = make_xic_array(
            ms2_peak_df,
            frame_num_map,
            ms2_st,
            ms2_ed,
            min_frame_index,
            max_frame_index,
            num_fragments,
        )

        ## count XIC peaks across fragments
        xic_peak_count_arr = count_xic_peaks(xic_arr)
        ## count mono-isotopic peaks of fragments
        spec_peak_count_arr = (xic_arr > 0).sum(axis=0, dtype=np.uint16)
        quant_peak_count_arr = (xic_arr[-QUANT_FRAGMENTS:, :] > 0).sum(
            axis=0, dtype=np.uint16
        )
        mask = (xic_peak_count_arr >= min_xic_peak_count) & (quant_peak_count_arr > 0)
        peak_group_arr = np.flatnonzero(mask).astype(np.uint32)

        if peak_group_arr.shape[0] < 1:
            continue

        ## cluster peak groups with ad-hoc peak group scoring
        comb_count_arr = xic_peak_count_arr + spec_peak_count_arr
        peak_group_score1_arr = (
            0.25 * comb_count_arr[peak_group_arr - 1]
            + 0.5 * comb_count_arr[peak_group_arr]
            + 0.25 * comb_count_arr[peak_group_arr + 1]
        )
        # peak_group_weights = (
        #     xic_peak_count_arr[peak_group_arr] + spec_peak_count_arr[peak_group_arr]
        # )
        peak_group_arr = cluster_peaks_with_weights(
            peak_group_arr, peak_group_score1_arr, dist_cutoff=2, min_cluster_size=1
        )
        mask = (peak_group_arr >= peak_lb) & (peak_group_arr <= peak_ub)
        peak_group_arr = peak_group_arr[mask]

        ## count the number of experimental peaks for each peak group
        sum_peak_count_arr = sum_neighborhood(
            all_peak_count_arr, rt_window_radius, peak_group_arr
        )

        ## filtering with min_peak_count
        mask = sum_peak_count_arr >= min_peak_count
        sum_peak_count_arr = sum_peak_count_arr[mask]
        peak_group_arr = peak_group_arr[mask]

        if peak_group_arr.shape[0] < 1:
            continue

        ## select Top K peak groups with more sophisticated ad-hoc scoring
        if peak_group_arr.shape[0] > topk:
            # ad-hoc scoring
            peak_group_scores = (
                # averaged correlations
                get_avg_xic_corrs(xic_arr, peak_group_arr, rt_window_radius=2)
                # fragment mono-isotopic peaks
                + 0.25 * spec_peak_count_arr[peak_group_arr - 1]
                + 0.5 * spec_peak_count_arr[peak_group_arr]
                + 0.25 * spec_peak_count_arr[peak_group_arr + 1]
            )
            ii = np.argsort(peak_group_scores)[::-1][:topk]
            peak_group_arr = peak_group_arr[ii]
            sum_peak_count_arr = sum_peak_count_arr[ii]

        ## save results
        # peak_group_arr += min_frame_index
        n_groups = peak_group_arr.shape[0]
        for k in range(n_groups):
            peak_group_arr[k] += min_frame_index
        peak_group_results[precursor_index0] = peak_group_arr
        peak_count_results[precursor_index0] = sum_peak_count_arr
        num_peak_groups[precursor_index0] = n_groups

    num_peak_groups = np.cumsum(num_peak_groups)
    precursor_index0_arr = np.empty(num_peak_groups[-1], dtype=np.uint32)
    frame_num_arr = np.empty(num_peak_groups[-1], dtype=np.uint32)
    peak_count_arr = np.empty(num_peak_groups[-1], dtype=np.uint16)

    for i in nb.prange(num_precursors):
        st = 0 if i == 0 else num_peak_groups[i - 1]
        ed = num_peak_groups[i]
        if ed > st:
            precursor_index0_arr[st:ed] = i
            frame_num_arr[st:ed] = frame_num_map.ms2_frame_num_arr[
                peak_group_results[i]
            ]
            peak_count_arr[st:ed] = peak_count_results[i]

    peak_group_results.clear()
    peak_count_results.clear()

    peak_group_container = PeakGroupContainer(
        min_precursor_index, precursor_index0_arr, frame_num_arr, peak_count_arr
    )
    peak_index_container = PeakIndexContainer(
        precursor_ms1_indices,
        precursor_ms2_indices,
        fragment_ms2_indices,
    )
    return peak_group_container, peak_index_container
