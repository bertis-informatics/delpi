import numpy as np
import numba as nb
from numba.extending import register_jitable

from delpi.lcms.fragmentation import IonTypeContainer
from delpi.database.numba.prefix_mass_array import PrefixMassArrayContainer
from delpi.database.numba.spec_lib_container import (
    SpectralLibContainer,
    TheoreticalPeakContainer,
)
from delpi.constants import PROTON_MASS, C13C12_MASS_DIFF


@register_jitable  # This will run in JIT mode only if called from a JIT function
def set_seed_compat(x):
    np.random.seed(x)


@nb.njit(nogil=True, cache=True)
def get_frame_index_range(
    speclib_container: SpectralLibContainer,
    ms2_rt_arr: np.ndarray,
    precursor_index0: int,
):
    peptidoform_index = speclib_container.precursor_peptidoform_index_arr[
        precursor_index0
    ]
    rt_lb = speclib_container.mod_rt_lb_arr[peptidoform_index]
    rt_ub = speclib_container.mod_rt_ub_arr[peptidoform_index]

    if np.isnan(rt_lb) or np.isnan(rt_lb):
        return np.uint32(0), np.uint32(ms2_rt_arr.shape[0])

    min_frame_index = np.searchsorted(ms2_rt_arr, rt_lb, side="left")
    max_frame_index = np.searchsorted(ms2_rt_arr, rt_ub, side="right")

    return np.uint32(min_frame_index), np.uint32(max_frame_index)


@nb.njit(inline="always")
def get_averagine_start_index(
    speclib_container: SpectralLibContainer, precursor_index0: int
) -> int:

    precursor_mz = speclib_container.precursor_mz_arr[precursor_index0]
    precursor_charge = speclib_container.precursor_charge_arr[precursor_index0]
    precursor_mass = (precursor_mz - PROTON_MASS) * precursor_charge

    if precursor_mass < speclib_container.averagine_min_nominal_mass:
        nominal_mass = speclib_container.averagine_min_nominal_mass
    elif precursor_mass > speclib_container.averagine_max_nominal_mass:
        nominal_mass = speclib_container.averagine_max_nominal_mass
    else:
        nominal_mass = (precursor_mass // 100) * 100

    i = (nominal_mass - speclib_container.averagine_min_nominal_mass) // 100
    st = i * speclib_container.max_precursor_isotopes

    return np.uint32(st)


@nb.njit(nogil=True, cache=True)
def get_theoretical_peaks(
    speclib_container: SpectralLibContainer, precursor_index0: int
) -> TheoreticalPeakContainer:

    peptidoform_index = speclib_container.precursor_peptidoform_index_arr[
        precursor_index0
    ]
    peptide_index = speclib_container.mod_peptide_index_arr[peptidoform_index]
    seq_len = speclib_container.peptide_seq_len_arr[peptide_index]

    precursor_charge = speclib_container.precursor_charge_arr[precursor_index0]
    precursor_mz = speclib_container.precursor_mz_arr[precursor_index0]

    st = get_averagine_start_index(speclib_container, precursor_index0)
    ed = st + speclib_container.max_precursor_isotopes

    precursor_mz_arr = np.empty(ed - st, dtype=np.float32)
    for i, j in enumerate(range(st, ed)):
        precursor_mz_arr[i] = (
            precursor_mz
            + (speclib_container.averagine_isotope_index_arr[j] * C13C12_MASS_DIFF)
            / precursor_charge
        )
    precursor_intensity_arr = speclib_container.averagine_predicted_intensity[st:ed]
    precursor_isotope_index_arr = speclib_container.averagine_isotope_index_arr[st:ed]

    num_fragments = speclib_container.max_fragments
    max_isotopes = speclib_container.max_fragment_isotopes
    st = precursor_index0 * num_fragments
    ed = st + num_fragments

    # fragment_mz_arr = speclib_container.speclib_mz_arr[st:ed]
    fragment_intensity_arr = speclib_container.speclib_predicted_intensity_arr[st:ed]
    fragment_cleavage_index_arr = speclib_container.speclib_cleavage_index_arr[st:ed]
    fragment_is_prefix_arr = speclib_container.speclib_is_prefix_arr[st:ed]
    fragment_charge_arr = speclib_container.speclib_charge_arr[st:ed]

    fragment_mz_arr = np.empty((max_isotopes, num_fragments), dtype=np.float32)
    for i in range(max_isotopes):
        fragment_mz_arr[i, :] = speclib_container.speclib_mz_arr[st:ed]

    for iso_index in range(1, max_isotopes):
        for i in range(num_fragments):
            fragment_mz_arr[iso_index, i] += (
                C13C12_MASS_DIFF * iso_index
            ) / fragment_charge_arr[i]

    return TheoreticalPeakContainer(
        sequence_length=seq_len,
        precursor_mz_arr=precursor_mz_arr,
        precursor_intensity_arr=precursor_intensity_arr,
        precursor_isotope_index_arr=precursor_isotope_index_arr,
        precursor_charge=precursor_charge,
        fragment_mz_arr=fragment_mz_arr,
        fragment_charge_arr=fragment_charge_arr,
        fragment_is_prefix_arr=fragment_is_prefix_arr,
        fragment_cleavage_index_arr=fragment_cleavage_index_arr,
        fragment_intensity_arr=fragment_intensity_arr,
    )


@nb.njit(nogil=True, cache=True)
def _select_intense_fragments(
    ion_type_container: IonTypeContainer,
    prefix_mass_arr: np.ndarray,
    precursor_index: int,
    intensity_arr: np.ndarray,
    max_fragments: int = 16,
    detectable_min_mz: float = 200.0,
    detectable_max_mz: float = 2000.0,
):
    set_seed_compat(precursor_index)

    ion_count = ion_type_container.charge_arr.shape[0]
    cleavage_count = intensity_arr.shape[0]
    # ion_type_index_arr = np.tile(np.arange(ion_count, dtype=np.uint8), cleavage_count)
    # cleavage_index_arr = np.tile(np.arange(cleavage_count, dtype=np.uint8).reshape(-1, 1), (1, ion_count))
    ion_type_index_arr = np.array(
        [j for _ in range(cleavage_count) for j in range(ion_count)], dtype=np.uint8
    )
    cleavage_index_arr = np.array(
        [j for j in range(cleavage_count) for _ in range(ion_count)], dtype=np.uint8
    )

    mz_arr = np.empty(intensity_arr.shape, dtype=np.float32)
    for ion_type_idx in range(ion_type_container.charge_arr.shape[0]):
        is_prefix = ion_type_container.is_prefix_arr[ion_type_idx]
        charge = ion_type_container.charge_arr[ion_type_idx]
        offset_mass = ion_type_container.offset_mass_arr[ion_type_idx]
        frag_mass = (
            prefix_mass_arr[:-1]
            if is_prefix
            else prefix_mass_arr[-1] - prefix_mass_arr[0:-1]
        )
        frag_mz = ((frag_mass + offset_mass) / charge) + PROTON_MASS
        mz_arr[:, ion_type_idx] = frag_mz

    intensity_arr = intensity_arr.flatten()
    mz_arr = mz_arr.flatten()
    # cleavage_index_arr.flatten()

    undetectable_mask = (
        (mz_arr < detectable_min_mz)
        | (mz_arr > detectable_max_mz)
        | (intensity_arr < 1e-4)
    )
    intensity_arr[undetectable_mask] = 0
    zero_mask = intensity_arr == 0
    intensity_arr[zero_mask] = 1e-4 * np.random.rand(zero_mask.sum())
    sorted_ii = intensity_arr.argsort()[-max_fragments:]
    intensity_arr /= intensity_arr[sorted_ii[-1]]
    ion_type_arr = ion_type_index_arr[sorted_ii]

    return (
        cleavage_index_arr[sorted_ii],
        ion_type_container.is_prefix_arr[ion_type_arr],
        ion_type_container.charge_arr[ion_type_arr],
        mz_arr[sorted_ii],
        intensity_arr[sorted_ii],
    )


@nb.njit(parallel=True, cache=True)
def update_speclib_arr(
    out_precursor_index_arr: np.ndarray,
    out_clevage_index_arr: np.ndarray,
    out_is_prefix_arr: np.ndarray,
    out_charge_arr: np.ndarray,
    out_mz_arr: np.ndarray,
    out_pred_intensity_arr: np.ndarray,
    out_rank_arr: np.ndarray,
    prefix_mass_container: PrefixMassArrayContainer,
    ion_type_container: IonTypeContainer,
    peptidoform_index_arr: np.ndarray,
    batch_precursor_index_arr: np.ndarray,
    batch_intensity_arr: np.ndarray,
    max_fragments: int = 16,
    detectable_min_mz: float = 200.0,
    detectable_max_mz: float = 2000.0,
):

    ion_count = ion_type_container.charge_arr.shape[0]
    batch_intensity_arr = batch_intensity_arr[..., :ion_count]
    batch_count, cleavage_count, _ = batch_intensity_arr.shape
    prefix_mass_stop_idx_arr = prefix_mass_container.mass_array_stop_index

    for i in nb.prange(batch_count):
        intensity_arr = batch_intensity_arr[i, ...]
        precursor_index = batch_precursor_index_arr[i]
        peptidoform_index = peptidoform_index_arr[precursor_index]
        st = (
            0
            if peptidoform_index == 0
            else prefix_mass_stop_idx_arr[peptidoform_index - 1]
        )
        prefix_mass_arr = prefix_mass_container.prefix_mass_array[
            st : st + cleavage_count + 1
        ]

        cleavage_index_arr, is_prefix_arr, charge_arr, mz_arr, pred_intensity_arr = (
            _select_intense_fragments(
                ion_type_container,
                prefix_mass_arr,
                precursor_index,
                intensity_arr,
                max_fragments,
                detectable_min_mz,
                detectable_max_mz,
            )
        )

        # update speclib arrays
        st = precursor_index * max_fragments
        ed = st + max_fragments
        out_precursor_index_arr[st:ed] = precursor_index
        out_clevage_index_arr[st:ed] = cleavage_index_arr
        out_is_prefix_arr[st:ed] = is_prefix_arr
        out_charge_arr[st:ed] = charge_arr
        out_mz_arr[st:ed] = mz_arr
        out_pred_intensity_arr[st:ed] = pred_intensity_arr
        for k in range(max_fragments):
            out_rank_arr[st + k] = max_fragments - k
