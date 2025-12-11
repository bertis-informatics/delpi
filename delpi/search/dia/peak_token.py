import numpy as np
import numba as nb

from delpi.lcms.data_container import DIAWindowFrameNumMap, PeakContainer
from delpi.database.numba.spec_lib_container import TheoreticalPeakContainer
from delpi.model.input import TheoPeakInput, ExpPeakInput
from delpi.search.dia.peak_group import PeakIndexContainer, QUANT_FRAGMENTS
from delpi.utils.peak import find_peak_index


MAX_EXP_PEAK_TOKENS = 512

THEO_TOKEN_DIM = len(TheoPeakInput)
EXP_TOKEN_DIM = len(ExpPeakInput)

THEO_PRED_INTENSITY_IDX = TheoPeakInput.PRED_INTENSITY.index
THEO_MZ_IDX = TheoPeakInput.MZ.index
THEO_IS_PRECURSOR_IDX = TheoPeakInput.IS_PRECURSOR.index
THEO_IS_PREFIX_IDX = TheoPeakInput.IS_PREFIX.index
THEO_CHARGE_IDX = TheoPeakInput.CHARGE.index
THEO_ISOTOPE_INDEX_IDX = TheoPeakInput.ISOTOPE_INDEX.index
THEO_CLEAVAGE_INDEX_IDX = TheoPeakInput.CLEAVAGE_INDEX.index
THEO_REV_CLEAVAGE_INDEX_IDX = TheoPeakInput.REV_CLEAVAGE_INDEX.index

EXP_MZ_ERROR_IDX = ExpPeakInput.MZ_ERROR.index
EXP_AB_IDX = ExpPeakInput.AB.index
EXP_Z_SCORE_IDX = ExpPeakInput.Z_SCORE.index
EXP_IS_PRECURSOR_IDX = ExpPeakInput.IS_PRECURSOR.index
EXP_IS_PREFIX_IDX = ExpPeakInput.IS_PREFIX.index
EXP_CHARGE_IDX = ExpPeakInput.CHARGE.index
EXP_ISOTOPE_INDEX_IDX = ExpPeakInput.ISOTOPE_INDEX.index
EXP_MS_LEVEL_IDX = ExpPeakInput.MS_LEVEL.index
EXP_CLEAVAGE_INDEX_IDX = ExpPeakInput.CLEAVAGE_INDEX.index
EXP_REV_CLEAVAGE_INDEX_IDX = ExpPeakInput.REV_CLEAVAGE_INDEX.index
EXP_TIME_INDEX_IDX = ExpPeakInput.TIME_INDEX.index

QUANT_AB_IDX = 0
QUANT_THEO_IDX = 1
QUANT_TIME_IDX = 2


@nb.njit(nogil=True, cache=True)
def get_x_theo(theo_peaks: TheoreticalPeakContainer, x_theo: np.ndarray = None):
    num_theo_peaks = (
        theo_peaks.precursor_mz_arr.shape[0] + theo_peaks.fragment_charge_arr.shape[0]
    )

    if x_theo is None:
        x_theo = np.empty((num_theo_peaks, THEO_TOKEN_DIM), dtype=np.float32)

    m = theo_peaks.precursor_mz_arr.shape[0]

    # precursor peaks
    x_theo[:m, THEO_PRED_INTENSITY_IDX] = theo_peaks.precursor_intensity_arr
    x_theo[:m, THEO_MZ_IDX] = theo_peaks.precursor_mz_arr
    x_theo[:m, THEO_IS_PRECURSOR_IDX] = 1.0
    x_theo[:m, THEO_IS_PREFIX_IDX] = 0.0
    x_theo[:m, THEO_CHARGE_IDX] = nb.float32(theo_peaks.precursor_charge)
    x_theo[:m, THEO_ISOTOPE_INDEX_IDX] = theo_peaks.precursor_isotope_index_arr
    x_theo[:m, THEO_CLEAVAGE_INDEX_IDX] = 0.0
    x_theo[:m, THEO_REV_CLEAVAGE_INDEX_IDX] = 0.0

    # fragment peaks

    seq_len = theo_peaks.sequence_length
    cleavage_index_arr = theo_peaks.fragment_cleavage_index_arr

    x_theo[m:, THEO_PRED_INTENSITY_IDX] = theo_peaks.fragment_intensity_arr
    x_theo[m:, THEO_MZ_IDX] = theo_peaks.fragment_mz_arr[0]  # only monoisotopic peaks
    x_theo[m:, THEO_IS_PRECURSOR_IDX] = 0.0
    x_theo[m:, THEO_IS_PREFIX_IDX] = theo_peaks.fragment_is_prefix_arr
    x_theo[m:, THEO_CHARGE_IDX] = theo_peaks.fragment_charge_arr
    x_theo[m:, THEO_ISOTOPE_INDEX_IDX] = 0.0
    x_theo[m:, THEO_CLEAVAGE_INDEX_IDX] = cleavage_index_arr
    x_theo[m:, THEO_REV_CLEAVAGE_INDEX_IDX] = seq_len - 2 - cleavage_index_arr

    return x_theo


@nb.njit(nogil=True, fastmath=True, cache=True)
def _set_x_exp(
    theo_peaks: TheoreticalPeakContainer,
    is_precursor: bool,
    ms_level: int,
    peak_df: PeakContainer,
    st_indices: np.ndarray,
    ed_indices: np.ndarray,
    min_frame_index: int,
    max_frame_index: int,
    frame_num_map: DIAWindowFrameNumMap,
    mz_tol: float,
    x_exp: np.ndarray,
    counter: int,
    x_ind: np.ndarray = None,
    x_quant: np.ndarray = None,
):
    if is_precursor:
        mz_arr = theo_peaks.precursor_mz_arr
    else:
        mz_arr = theo_peaks.fragment_mz_arr.flatten()

    num_fragments = theo_peaks.fragment_charge_arr.shape[0]
    seq_len = theo_peaks.sequence_length
    start_counter = counter
    ind_count = 0
    quant_peak_count = 0
    max_ab = 0.0
    for theo_i_, (st, ed) in enumerate(zip(st_indices, ed_indices)):
        theo_mz = mz_arr[theo_i_]
        if is_precursor:
            theo_i = theo_i_
            theo_charge = theo_peaks.precursor_charge
            theo_isotope_index = theo_peaks.precursor_isotope_index_arr[theo_i]
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
            fi = frame_num_map.frame_num_to_index_arr[fn]
            if (fi >= min_frame_index) and (fi <= max_frame_index):
                mz = peak_df.mz_arr[exp_i]
                ab = peak_df.ab_arr[exp_i]
                z_score = peak_df.z_score_arr[exp_i]
                time_index = fi - min_frame_index
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
                x_exp[counter, EXP_TIME_INDEX_IDX] = time_index

                if (
                    (x_quant is not None)
                    and (fi > min_frame_index)
                    and (fi < max_frame_index)
                    and (theo_isotope_index == 0)
                    and (
                        theo_i > (num_fragments - QUANT_FRAGMENTS - 1)
                    )  # skip first n fragments because fragments are sorted by increasing intensity
                    and (quant_peak_count < x_quant.shape[0])
                ):
                    x_quant[quant_peak_count, QUANT_AB_IDX] = ab
                    x_quant[quant_peak_count, QUANT_THEO_IDX] = theo_i
                    x_quant[quant_peak_count, QUANT_TIME_IDX] = time_index
                    quant_peak_count += 1

                # save matched peak indices at the RT center
                if (
                    (x_ind is not None)
                    and (fi > min_frame_index + 2)
                    and (fi < max_frame_index - 2)
                    and (ind_count < x_ind.shape[0])
                ):
                    x_ind[ind_count] = exp_i
                    ind_count += 1
                counter += 1

    # abundance scaling
    if max_ab > 0:
        x_exp[start_counter:counter, EXP_AB_IDX] /= max_ab

    return counter, max_ab


@nb.njit(nogil=True, cache=True)
def get_x_exp(
    precursor_index0: int,
    frame_num: int,
    theo_peaks: TheoreticalPeakContainer,
    ms1_peak_df: PeakContainer,
    ms2_peak_df: PeakContainer,
    frame_num_map: DIAWindowFrameNumMap,
    rt_window_radius: int = 4,
    peak_index_container: PeakIndexContainer = None,
    ms1_mass_tol: float = 10,
    ms2_mass_tol: float = 10,
    x_exp: np.ndarray = None,
    x_ind: np.ndarray = None,
    x_quant: np.ndarray = None,
):
    # precursor_index0 = precursor_index - frag_db.min_precursor_index
    frame_index = frame_num_map.frame_num_to_index_arr[frame_num]
    min_frame_index = frame_index - rt_window_radius
    max_frame_index = frame_index + rt_window_radius

    if x_exp is None:
        x_exp = np.empty(
            (512, EXP_TOKEN_DIM),
            dtype=np.float32,
        )

    if peak_index_container is None:
        # searching for precursor ion peaks in MS1 & MS2
        mz_arr = theo_peaks.precursor_mz_arr
        pc_ms1_st, pc_ms1_ed = find_peak_index(ms1_peak_df.mz_arr, mz_arr, ms1_mass_tol)
        pc_ms2_st, pc_ms2_ed = find_peak_index(ms2_peak_df.mz_arr, mz_arr, ms2_mass_tol)
        # searching for fragment ion peaks in MS2
        mz_arr = theo_peaks.fragment_mz_arr
        ms2_st, ms2_ed = find_peak_index(
            ms2_peak_df.mz_arr, mz_arr.flatten(), ms2_mass_tol
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
        st_indices=pc_ms1_st,
        ed_indices=pc_ms1_ed,
        min_frame_index=min_frame_index,
        max_frame_index=max_frame_index,
        frame_num_map=frame_num_map,
        mz_tol=ms1_mass_tol,
        x_exp=x_exp,
        counter=0,
    )

    counter, _ = _set_x_exp(
        theo_peaks,
        is_precursor=True,
        ms_level=2,
        peak_df=ms2_peak_df,
        st_indices=pc_ms2_st,
        ed_indices=pc_ms2_ed,
        min_frame_index=min_frame_index,
        max_frame_index=max_frame_index,
        frame_num_map=frame_num_map,
        mz_tol=ms2_mass_tol,
        x_exp=x_exp,
        counter=counter,
    )

    counter, _ = _set_x_exp(
        theo_peaks,
        is_precursor=False,
        ms_level=2,
        peak_df=ms2_peak_df,
        st_indices=ms2_st,
        ed_indices=ms2_ed,
        min_frame_index=min_frame_index,
        max_frame_index=max_frame_index,
        frame_num_map=frame_num_map,
        mz_tol=ms2_mass_tol,
        x_exp=x_exp,
        counter=counter,
        x_ind=x_ind,
        x_quant=x_quant,
    )

    return x_exp[:counter], ms1_scale
