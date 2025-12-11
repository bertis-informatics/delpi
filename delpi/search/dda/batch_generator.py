from typing import Iterator, Tuple

import numba as nb
import numpy as np


from delpi.database.numba.spec_lib_container import SpectralLibContainer
from delpi.lcms.data_container import PeakContainer, MetaContainer
from delpi.search.dda.peak_group import DdaPeakGroupContainer
from delpi.search.dia.peak_group import PeakIndexContainer
from delpi.search.dia.batch_generator import count_total_batches, iter_batch_indices
from delpi.database.numba.spec_lib_utils import get_theoretical_peaks
from delpi.search.dia.peak_token import get_x_theo, MAX_EXP_PEAK_TOKENS
from delpi.search.dda.peak_token import (
    get_x_exp,
    EXP_TOKEN_DIM,
    THEO_TOKEN_DIM,
)


@nb.njit(parallel=True, cache=True)
def _make_batch_in_parallel(
    batch_indices: np.ndarray,
    speclib_container: SpectralLibContainer,
    ms1_meta_df: MetaContainer,
    ms2_meta_df: MetaContainer,
    ms1_peak_df: PeakContainer,
    ms2_peak_df: PeakContainer,
    peak_group_container: DdaPeakGroupContainer,
    peak_index_container: PeakIndexContainer,
    ms1_mass_tol: float,
    ms2_mass_tol: float,
    X_precursor_index: np.ndarray,
    X_theo: np.ndarray,
    X_exp: np.ndarray,
    X_indices: np.ndarray,
    ms1_scale_arr: np.ndarray,
):
    precursor_index0_arr = peak_group_container.precursor_index0_arr
    min_precursor_index = speclib_container.min_precursor_index

    cur_batch_size = batch_indices.shape[0]
    X_indices[:] = -1
    ms1_scale_arr[:] = -1.0

    for i in nb.prange(cur_batch_size):
        k = batch_indices[i]
        precursor_index0 = precursor_index0_arr[k]
        ms2_frame_num = peak_group_container.frame_num_arr[k]

        theo_peaks = get_theoretical_peaks(speclib_container, precursor_index0)
        _ = get_x_theo(theo_peaks, X_theo[i])

        _, ms1_scale = get_x_exp(
            precursor_index0=precursor_index0,
            ms2_frame_num=ms2_frame_num,
            theo_peaks=theo_peaks,
            ms1_meta_df=ms1_meta_df,
            ms2_meta_df=ms2_meta_df,
            ms1_peak_df=ms1_peak_df,
            ms2_peak_df=ms2_peak_df,
            frame_num_key_arr=peak_group_container.frame_num_key_arr[k],
            time_index_val_arr=peak_group_container.time_index_val_arr[k],
            ms1_mass_tol=ms1_mass_tol,
            ms2_mass_tol=ms2_mass_tol,
            peak_index_container=peak_index_container,
            x_exp=X_exp[i],
            x_ind=X_indices[i],
        )
        X_precursor_index[i] = precursor_index0 + min_precursor_index
        ms1_scale_arr[i] = ms1_scale


def generate_batches(
    speclib_container: SpectralLibContainer,
    ms1_meta_df: MetaContainer,
    ms2_meta_df: MetaContainer,
    ms1_peak_df: PeakContainer,
    ms2_peak_df: PeakContainer,
    peak_group_container: DdaPeakGroupContainer,
    peak_index_container: PeakIndexContainer,
    ms1_mass_tol: float,
    ms2_mass_tol: float,
    batch_size: int,
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
    X_indices = np.empty((batch_size, 64), dtype=np.int32)

    frame_num_arr = peak_group_container.frame_num_arr
    # precursor_index0_arr = peak_group_container.precursor_index0_arr
    # min_precursor_index = frag_db.min_precursor_index
    batch_iter = iter_batch_indices(peak_group_container.peak_count_arr, batch_size)

    for num_peaks, batch_indices in batch_iter:
        cur_batch_size = batch_indices.shape[0]
        _make_batch_in_parallel(
            batch_indices,
            speclib_container,
            ms1_meta_df,
            ms2_meta_df,
            ms1_peak_df,
            ms2_peak_df,
            peak_group_container,
            peak_index_container,
            ms1_mass_tol,
            ms2_mass_tol,
            X_precursor_index,
            X_theo,
            X_exp,
            X_indices,
            ms1_scale_arr,
        )

        yield (
            X_precursor_index[:cur_batch_size],
            frame_num_arr[batch_indices],
            X_theo[:cur_batch_size, :, :],
            X_exp[:cur_batch_size, :num_peaks, :],
            X_indices[:cur_batch_size, :],
            ms1_scale_arr[:cur_batch_size],
        )
