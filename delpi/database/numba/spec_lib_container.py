from typing import NamedTuple

import numpy as np


class SpectralLibContainer(NamedTuple):

    min_precursor_index: int
    max_precursor_index: int
    max_fragments: int
    max_precursor_isotopes: int
    max_fragment_isotopes: int

    peptide_is_decoy_arr: np.ndarray
    peptide_seq_len_arr: np.ndarray

    mod_peptide_index_arr: np.ndarray
    mod_ref_rt_arr: np.ndarray
    mod_predicted_rt_arr: np.ndarray
    mod_rt_lb_arr: np.ndarray
    mod_rt_ub_arr: np.ndarray

    precursor_peptidoform_index_arr: np.ndarray
    precursor_mz_arr: np.ndarray
    precursor_charge_arr: np.ndarray

    speclib_cleavage_index_arr: np.ndarray
    speclib_is_prefix_arr: np.ndarray
    speclib_charge_arr: np.ndarray
    speclib_mz_arr: np.ndarray
    speclib_predicted_intensity_arr: np.ndarray

    averagine_min_nominal_mass: int
    averagine_max_nominal_mass: int
    averagine_predicted_intensity: np.ndarray
    averagine_isotope_index_arr: np.ndarray


class TheoreticalPeakContainer(NamedTuple):

    sequence_length: int
    precursor_mz_arr: np.ndarray
    precursor_intensity_arr: np.ndarray
    precursor_isotope_index_arr: np.ndarray
    precursor_charge: int

    fragment_mz_arr: np.ndarray
    fragment_charge_arr: np.ndarray
    fragment_is_prefix_arr: np.ndarray
    fragment_cleavage_index_arr: np.ndarray
    fragment_intensity_arr: np.ndarray
