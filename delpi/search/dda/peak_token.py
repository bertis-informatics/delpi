import numpy as np
import numba as nb

from delpi.lcms.data_container import PeakContainer, MetaContainer
from delpi.database.numba.spec_lib_container import TheoreticalPeakContainer
from delpi.search.dia.peak_group import PeakIndexContainer
from delpi.search.dda.peak_group import lookup_time_index
from delpi.utils.peak import find_peak_index
from delpi.search.dia.peak_token import (
    MAX_EXP_PEAK_TOKENS,
    THEO_TOKEN_DIM,
    EXP_TOKEN_DIM,
    EXP_MZ_ERROR_IDX,
    EXP_AB_IDX,
    EXP_Z_SCORE_IDX,
    EXP_IS_PRECURSOR_IDX,
    EXP_IS_PREFIX_IDX,
    EXP_CHARGE_IDX,
    EXP_ISOTOPE_INDEX_IDX,
    EXP_MS_LEVEL_IDX,
    EXP_CLEAVAGE_INDEX_IDX,
    EXP_REV_CLEAVAGE_INDEX_IDX,
    EXP_TIME_INDEX_IDX,
)


@nb.njit(nogil=True, fastmath=True, cache=True)
def _set_x_exp(
    theo_peaks: TheoreticalPeakContainer,
    is_precursor: bool,
    ms_level: int,
    peak_df: PeakContainer,
    meta_df: MetaContainer,
    st_indices: np.ndarray,
    ed_indices: np.ndarray,
    frame_num_key_arr: np.ndarray,
    time_index_val_arr: np.ndarray,
    mz_tol: float,
    x_exp: np.ndarray,
    counter: int,
    ms2_frame_num: int,
    x_ind: np.ndarray = None,
):
    if is_precursor:
        mz_arr = theo_peaks.precursor_mz_arr
    else:
        mz_arr = theo_peaks.fragment_mz_arr.flatten()

    num_fragments = theo_peaks.fragment_charge_arr.shape[0]
    seq_len = theo_peaks.sequence_length
    start_counter = counter
    ind_count = 0
    max_ab = 0.0
    for theo_i_, (st, ed) in enumerate(zip(st_indices, ed_indices)):
        theo_mz = mz_arr[theo_i_]
        if is_precursor:
            theo_charge = theo_peaks.precursor_charge
            theo_isotope_index = theo_peaks.precursor_isotope_index_arr[theo_i_]
            theo_is_prefix = False
            theo_cleavage_index = 0.0
            theo_rev_cleavage_index = 0.0
        else:
            theo_i = theo_i_ % num_fragments
            theo_isotope_index = theo_i_ // num_fragments
            theo_charge = theo_peaks.fragment_charge_arr[theo_i]
            theo_is_prefix = theo_peaks.fragment_is_prefix_arr[theo_i]
            theo_cleavage_index = theo_peaks.fragment_cleavage_index_arr[theo_i]
            theo_rev_cleavage_index = seq_len - 2 - theo_cleavage_index

        for exp_i in range(st, ed):
            fn = peak_df.frame_num_arr[exp_i]
            time_idx = lookup_time_index(frame_num_key_arr, time_index_val_arr, fn)

            if time_idx >= 0:
                mz = peak_df.mz_arr[exp_i]
                ab = peak_df.ab_arr[exp_i]
                z_score = peak_df.z_score_arr[exp_i]
                max_ab = max(ab, max_ab)
                x_exp[counter, EXP_MZ_ERROR_IDX] = (
                    (mz - theo_mz) / theo_mz * (1e6 / mz_tol)
                )
                x_exp[counter, EXP_AB_IDX] = ab
                x_exp[counter, EXP_Z_SCORE_IDX] = z_score
                x_exp[counter, EXP_IS_PRECURSOR_IDX] = is_precursor
                x_exp[counter, EXP_IS_PREFIX_IDX] = theo_is_prefix
                x_exp[counter, EXP_CHARGE_IDX] = theo_charge
                x_exp[counter, EXP_ISOTOPE_INDEX_IDX] = theo_isotope_index
                x_exp[counter, EXP_MS_LEVEL_IDX] = ms_level
                x_exp[counter, EXP_CLEAVAGE_INDEX_IDX] = theo_cleavage_index
                x_exp[counter, EXP_REV_CLEAVAGE_INDEX_IDX] = theo_rev_cleavage_index
                x_exp[counter, EXP_TIME_INDEX_IDX] = time_idx

                # save matched peak indices at the RT center
                if (
                    (x_ind is not None)
                    and (fn == ms2_frame_num)
                    and (ind_count < x_ind.shape[0])
                ):
                    x_ind[ind_count] = exp_i
                    ind_count += 1

                counter += 1
                if counter >= x_exp.shape[0]:
                    print("Warning: MAX_EXP_PEAK_TOKENS exceeded.")
                    break

    # abundance scaling
    if max_ab > 0:
        x_exp[start_counter:counter, EXP_AB_IDX] /= max_ab

    return counter, max_ab


@nb.njit(nogil=True, fastmath=True, cache=True)
def get_x_exp(
    precursor_index0: int,
    ms2_frame_num: int,
    theo_peaks: TheoreticalPeakContainer,
    ms1_meta_df: MetaContainer,
    ms2_meta_df: MetaContainer,
    ms1_peak_df: PeakContainer,
    ms2_peak_df: PeakContainer,
    frame_num_key_arr: np.ndarray,
    time_index_val_arr: np.ndarray,
    ms1_mass_tol: float,
    ms2_mass_tol: float,
    peak_index_container: PeakIndexContainer = None,
    x_exp: np.ndarray = None,
    x_ind: np.ndarray = None,
):
    if x_exp is None:
        x_exp = np.empty(
            (MAX_EXP_PEAK_TOKENS, EXP_TOKEN_DIM),
            dtype=np.float32,
        )

    if peak_index_container is None:
        # searching for precursor ion peaks in MS1 & MS2
        mz_arr = theo_peaks.precursor_mz_arr
        pc_ms1_st, pc_ms1_ed = find_peak_index(ms1_peak_df.mz_arr, mz_arr, ms1_mass_tol)
        pc_ms2_st, pc_ms2_ed = find_peak_index(ms2_peak_df.mz_arr, mz_arr, ms2_mass_tol)
        # searching for fragment ion peaks in MS2
        ms2_st, ms2_ed = find_peak_index(
            ms2_peak_df.mz_arr, theo_peaks.fragment_mz_arr.flatten(), ms2_mass_tol
        )
    else:
        ms2_st, ms2_ed = peak_index_container.fragment_ms2_indices[precursor_index0]
        pc_ms1_st, pc_ms1_ed = peak_index_container.precursor_ms1_indices[
            precursor_index0
        ]
        pc_ms2_st, pc_ms2_ed = peak_index_container.precursor_ms2_indices[
            precursor_index0
        ]

    counter, ms1_scale = _set_x_exp(
        theo_peaks,
        is_precursor=True,
        ms_level=1,
        peak_df=ms1_peak_df,
        meta_df=ms1_meta_df,
        st_indices=pc_ms1_st,
        ed_indices=pc_ms1_ed,
        frame_num_key_arr=frame_num_key_arr[0],
        time_index_val_arr=time_index_val_arr[0],
        mz_tol=ms1_mass_tol,
        x_exp=x_exp,
        counter=0,
        ms2_frame_num=ms2_frame_num,
    )
    counter, _ = _set_x_exp(
        theo_peaks,
        is_precursor=True,
        ms_level=2,
        peak_df=ms2_peak_df,
        meta_df=ms2_meta_df,
        st_indices=pc_ms2_st,
        ed_indices=pc_ms2_ed,
        frame_num_key_arr=frame_num_key_arr[1],
        time_index_val_arr=time_index_val_arr[1],
        mz_tol=ms2_mass_tol,
        x_exp=x_exp,
        counter=counter,
        ms2_frame_num=ms2_frame_num,
    )
    counter, _ = _set_x_exp(
        theo_peaks,
        is_precursor=False,
        ms_level=2,
        peak_df=ms2_peak_df,
        meta_df=ms2_meta_df,
        st_indices=ms2_st,
        ed_indices=ms2_ed,
        frame_num_key_arr=frame_num_key_arr[1],
        time_index_val_arr=time_index_val_arr[1],
        mz_tol=ms2_mass_tol,
        x_exp=x_exp,
        counter=counter,
        ms2_frame_num=ms2_frame_num,
        x_ind=x_ind,
    )

    return x_exp[:counter], ms1_scale
