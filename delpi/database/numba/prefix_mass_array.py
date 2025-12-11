from typing import NamedTuple

import numba as nb
import numpy as np

from delpi.chem.amino_acid import AminoAcid
from delpi.chem.modification import Modification


aa_mass_array = np.zeros(128, dtype=np.float64)
for aa in AminoAcid.standard_amino_acids:
    aa_mass_array[ord(aa.residue)] = aa.mass

max_mod_id = Modification.get_max_accession_num()
mod_mass_array = np.zeros(max_mod_id + 1, dtype=np.float64)
for mod in Modification.name_to_mod_map.values():
    mod_mass_array[mod.accession_num] = mod.mass


@nb.njit(nogil=True, cache=True)
def get_prefix_mass_array(seq, mod_ids, mod_sites):
    mass_arr = np.zeros(len(seq) - 2, dtype=np.float32)
    for i, aa in enumerate(seq[1:-1]):
        mass_arr[i] = aa_mass_array[ord(aa)]

    for mod_id, site in zip(mod_ids, mod_sites):
        # site =  0 or -1           if n-term or c-term
        #         1-len(mass_arr)   otherwise
        mass_arr_pos = site - 1 if site > 0 else site
        unimod_id = (mod_id >> 16) & 0xFFFF
        mass_arr[mass_arr_pos] += mod_mass_array[unimod_id]

    return np.cumsum(mass_arr)


@nb.njit(parallel=True, cache=True)
def generate_prefix_mass_arrays(peptide_list, index_arr, mod_info_arr):
    n = index_arr.shape[0]

    # stop_indexes = np.cumsum(
    #     np.array([len(s) - 2 for s in peptides[index_arr[:, 0]]], dtype=np.uint32)
    # )
    stop_indexes = np.cumsum(
        np.array([len(peptide_list[i]) - 2 for i in index_arr[:, 0]], dtype=np.uint32)
    )

    flat_mass_arr = np.empty(stop_indexes[-1], dtype=np.float32)

    for i in nb.prange(n):
        pi = index_arr[i, 0]
        seq = peptide_list[pi]

        mod_start = index_arr[i, 1]
        mod_stop = index_arr[i, 2]

        mod_ids = mod_info_arr[mod_start:mod_stop, 0]
        mod_sites = mod_info_arr[mod_start:mod_stop, 1]

        prefix_mass_arr = get_prefix_mass_array(str(seq), mod_ids, mod_sites)
        mass_arr_start = 0 if i == 0 else stop_indexes[i - 1]
        mass_arr_stop = stop_indexes[i]
        flat_mass_arr[mass_arr_start:mass_arr_stop] = prefix_mass_arr
        # results[i] = prefix_mass_arr
    # return results
    return flat_mass_arr, stop_indexes


@nb.njit(parallel=True, cache=True)
def get_fragment_mass(
    flatten_prefix_mass_array: np.ndarray,
    peptidoform_index: np.ndarray,
    cleavage_index: np.ndarray,
    is_prefix: np.ndarray,
    peptidoform_mass_array_stop_index: np.ndarray,
):

    assert peptidoform_index.shape[0] == cleavage_index.shape[0]
    assert peptidoform_index.shape[0] == is_prefix.shape[0]

    n = peptidoform_index.shape[0]
    fragment_mass = np.zeros(n, dtype=np.float32)

    for i in nb.prange(n):
        p_idx = peptidoform_index[i]
        clvg_idx = cleavage_index[i]

        st = 0 if p_idx == 0 else peptidoform_mass_array_stop_index[p_idx - 1]
        prefix_mass = flatten_prefix_mass_array[st + clvg_idx]
        if is_prefix[i]:
            fragment_mass[i] = prefix_mass
        else:
            last = peptidoform_mass_array_stop_index[p_idx] - 1
            fragment_mass[i] = flatten_prefix_mass_array[last] - prefix_mass

    return fragment_mass


class PrefixMassArrayContainer(NamedTuple):
    prefix_mass_array: np.ndarray
    mass_array_stop_index: np.ndarray

    def _get_st(self, peptidoform_index):
        return (
            0
            if peptidoform_index == 0
            else self.mass_array_stop_index[peptidoform_index - 1]
        )

    def get_prefix_mass_array(self, peptidoform_index: int):

        st = self._get_st(peptidoform_index)
        ed = self.mass_array_stop_index[peptidoform_index]
        return self.prefix_mass_array[st:ed]

    def get_fragment_mass(self, peptidoform_index, cleavage_index, is_prefix):

        st = self._get_st(peptidoform_index)
        prefix_mass = self.prefix_mass_array[st + cleavage_index]
        if is_prefix:
            return prefix_mass
        last = self.mass_array_stop_index[peptidoform_index] - 1
        return self.prefix_mass_array[last] - prefix_mass
