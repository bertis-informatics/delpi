import numpy as np
import numba as nb

from delpi.lcms.data_container import IonTypeContainer
from delpi.utils.peak import find_peak_index
from delpi.constants import PROTON_MASS


@nb.njit(nogil=True, fastmath=True, cache=True)
def get_mz_arr(prefix_mass_arr: np.ndarray, ion_type_container: IonTypeContainer):
    num_cleavages = prefix_mass_arr.shape[0] - 1
    mz_arr = np.empty(
        (num_cleavages, ion_type_container.charge_arr.shape[0]), dtype=np.float32
    )
    # ["b_z1", "b_z2", "y_z1", "y_z2"]
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

    return mz_arr


@nb.njit(nogil=True, fastmath=True, cache=True)
def get_intensity_arr(
    mz_arr: np.ndarray, peak_arr: np.ndarray, tolerance_in_ppm: float
):

    start_indices, stop_indices = find_peak_index(
        peak_arr[:, 0], mz_arr.flatten(), tolerance_in_ppm
    )

    intensity_arr = np.empty(start_indices.shape, dtype=np.float32)
    max_intensity = 1e-12
    for i, (st, ed) in enumerate(zip(start_indices, stop_indices)):
        if st < ed:
            ab = peak_arr[st:ed, 1].max()
        else:
            ab = 0
        intensity_arr[i] = ab
        max_intensity = max(max_intensity, ab)
    intensity_arr /= max_intensity

    return intensity_arr.reshape(mz_arr.shape)
