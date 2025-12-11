from typing import NamedTuple

import numpy as np


class IonTypeContainer(NamedTuple):
    is_prefix_arr: np.ndarray
    charge_arr: np.ndarray
    offset_mass_arr: np.ndarray
    is_modloss_arr: np.ndarray


class PeakContainer(NamedTuple):
    frame_num_arr: np.ndarray
    ab_arr: np.ndarray
    mz_arr: np.ndarray
    z_score_arr: np.ndarray


class DIAWindowFrameNumMap(NamedTuple):
    ms1_frame_num_arr: np.ndarray
    ms2_frame_num_arr: np.ndarray
    ms1_rt_arr: np.ndarray
    ms2_rt_arr: np.ndarray
    frame_num_to_index_arr: np.ndarray


class MetaContainer(NamedTuple):
    frame_num_arr: np.ndarray
    frame_num_to_index_arr: np.ndarray
    ms_level: int
    rt_arr: np.ndarray
    isolation_min_mz_arr: np.ndarray
    isolation_max_mz_arr: np.ndarray


class SpectrumContainer(NamedTuple):
    frame_num: int
    ab_arr: np.ndarray
    mz_arr: np.ndarray
    z_score_arr: np.ndarray
