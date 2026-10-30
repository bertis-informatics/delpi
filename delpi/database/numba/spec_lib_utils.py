import numpy as np
import numba as nb

from delpi.lcms.data_container import IonTypeContainer
from delpi.database.numba.prefix_mass_array import PrefixMassArrayContainer
from delpi.database.numba.spec_lib_container import (
    SpectralLibContainer,
    TheoreticalPeakContainer,
)
from delpi.constants import PROTON_MASS, C13C12_MASS_DIFF


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

    if np.isnan(rt_lb) or np.isnan(rt_ub):
        return 0, ms2_rt_arr.shape[0] - 1

    min_frame_index = np.searchsorted(ms2_rt_arr, rt_lb, side="left")
    max_frame_index = min(
        np.searchsorted(ms2_rt_arr, rt_ub, side="right") - 1, ms2_rt_arr.shape[0] - 1
    )

    return min_frame_index, max_frame_index


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

    num_fragments = speclib_container.precursor_speclib_rows_arr[precursor_index0]
    max_isotopes = speclib_container.max_fragment_isotopes
    st = speclib_container.precursor_speclib_offset_arr[precursor_index0]
    ed = st + num_fragments

    # fragment_mz_arr = speclib_container.speclib_mz_arr[st:ed]
    fragment_intensity_arr = speclib_container.speclib_predicted_intensity_arr[st:ed]
    fragment_cleavage_index_arr = speclib_container.speclib_cleavage_index_arr[st:ed]
    fragment_is_prefix_arr = speclib_container.speclib_is_prefix_arr[st:ed]
    fragment_charge_arr = speclib_container.speclib_charge_arr[st:ed]

    fragment_isotope_index_arr = np.arange(-1, max_isotopes - 1, dtype=np.int8)
    fragment_mz_arr = np.empty((max_isotopes, num_fragments), dtype=np.float32)
    for iso_row in range(max_isotopes):
        for i in range(num_fragments):
            fragment_mz_arr[iso_row, i] = (
                speclib_container.speclib_mz_arr[st + i]
                + (C13C12_MASS_DIFF * fragment_isotope_index_arr[iso_row])
                / fragment_charge_arr[i]
            )

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
        fragment_is_modloss_arr=speclib_container.speclib_is_modloss_arr[st:ed],
        fragment_isotope_index_arr=fragment_isotope_index_arr,
    )


@nb.njit(nogil=True, cache=True)
def _select_intense_fragments(
    ion_type_container: IonTypeContainer,
    prefix_mass_arr: np.ndarray,
    intensity_arr: np.ndarray,
    max_fragments: int = 16,
    min_intensity: float = 0.01,
    detectable_min_mz: float = 200.0,
    detectable_max_mz: float = 2000.0,
):
    """Select the top N eligible fragments by predicted intensity.

    Normalize to the strongest finite, in-range fragment among the requested
    ion types before applying the inclusive threshold. Store selected rows in
    increasing intensity order.
    """
    ion_count = ion_type_container.charge_arr.shape[0]
    cleavage_count = intensity_arr.shape[0]
    assert intensity_arr.shape[1] == ion_count
    ion_type_index_arr = np.array(
        [j for _ in range(cleavage_count) for j in range(ion_count)], dtype=np.uint8
    )
    cleavage_index_arr = np.array(
        [j for j in range(cleavage_count) for _ in range(ion_count)], dtype=np.uint8
    )
    mz_arr = np.empty(intensity_arr.shape, dtype=np.float32)
    for ion_type_idx in range(ion_count):
        is_prefix = ion_type_container.is_prefix_arr[ion_type_idx]
        charge = ion_type_container.charge_arr[ion_type_idx]
        offset_mass = ion_type_container.offset_mass_arr[ion_type_idx]
        frag_mass = (
            prefix_mass_arr[:-1]
            if is_prefix
            else prefix_mass_arr[-1] - prefix_mass_arr[:-1]
        )
        mz_arr[:, ion_type_idx] = (frag_mass + offset_mass) / charge + PROTON_MASS

    intensity_arr = intensity_arr.flatten()
    mz_arr = mz_arr.flatten()
    candidates = np.flatnonzero(
        (mz_arr >= detectable_min_mz)
        & (mz_arr <= detectable_max_mz)
        & np.isfinite(mz_arr)
        & np.isfinite(intensity_arr)
        & (intensity_arr > 0)
    )
    if candidates.size:
        intensity_arr /= intensity_arr[candidates].max()
        candidates = candidates[intensity_arr[candidates] >= np.float32(min_intensity)]
    if candidates.size:
        # Stable tie-breaking is shared by in-memory and chunked prediction.
        is_prefix = ion_type_container.is_prefix_arr[ion_type_index_arr[candidates]]
        secondary = is_prefix.argsort(kind="mergesort")
        primary = intensity_arr[candidates][secondary].argsort(kind="mergesort")
        ordered = candidates[secondary][primary]
        selected = ordered[-max_fragments:]
    else:
        selected = np.empty(0, dtype=np.int64)
    ions = ion_type_index_arr[selected]
    return (
        cleavage_index_arr[selected],
        ion_type_container.is_prefix_arr[ions],
        ion_type_container.charge_arr[ions],
        ion_type_container.is_modloss_arr[ions],
        mz_arr[selected],
        intensity_arr[selected],
    )


@nb.njit(parallel=True, cache=True)
def update_speclib_chunk_arr(
    out_precursor_index_arr: np.ndarray,
    out_cleavage_index_arr: np.ndarray,
    out_is_prefix_arr: np.ndarray,
    out_charge_arr: np.ndarray,
    out_is_modloss_arr: np.ndarray,
    out_mz_arr: np.ndarray,
    out_pred_intensity_arr: np.ndarray,
    out_rank_arr: np.ndarray,
    out_num_rows_arr: np.ndarray,
    prefix_mass_container: PrefixMassArrayContainer,
    ion_type_container: IonTypeContainer,
    peptidoform_index_arr: np.ndarray,
    batch_precursor_index_arr: np.ndarray,
    batch_intensity_arr: np.ndarray,
    chunk_start_precursor_index: int,
    max_fragments: int = 16,
    min_intensity: float = 0.01,
    detectable_min_mz: float = 200.0,
    detectable_max_mz: float = 2000.0,
):
    """Write into temporary fixed-capacity slots; only actual rows are saved."""
    batch_count, cleavage_count, _ = batch_intensity_arr.shape
    stop_indices = prefix_mass_container.mass_array_stop_index
    for i in nb.prange(batch_count):
        precursor_index = batch_precursor_index_arr[i]
        peptidoform_index = peptidoform_index_arr[precursor_index]
        start = 0 if peptidoform_index == 0 else stop_indices[peptidoform_index - 1]
        prefix_mass_arr = prefix_mass_container.prefix_mass_array[
            start : start + cleavage_count + 1
        ]
        cleavage, is_prefix, charge, is_modloss, mz, intensity = (
            _select_intense_fragments(
                ion_type_container,
                prefix_mass_arr,
                batch_intensity_arr[i],
                max_fragments,
                min_intensity,
                detectable_min_mz,
                detectable_max_mz,
            )
        )
        local_index = np.int64(precursor_index) - np.int64(chunk_start_precursor_index)
        count = len(intensity)
        out_num_rows_arr[local_index] = count
        start = local_index * max_fragments
        stop = start + count
        out_precursor_index_arr[start:stop] = precursor_index
        out_cleavage_index_arr[start:stop] = cleavage
        out_is_prefix_arr[start:stop] = is_prefix
        out_charge_arr[start:stop] = charge
        out_is_modloss_arr[start:stop] = is_modloss
        out_mz_arr[start:stop] = mz
        out_pred_intensity_arr[start:stop] = intensity
        # Rank 1 is the highest predicted intensity, used by reader-side limits.
        for k in range(count):
            out_rank_arr[start + k] = count - k
