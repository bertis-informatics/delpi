from typing import List

import numpy as np
import numba as nb


@nb.njit(parallel=True, fastmath=True, cache=True)
def compute_z_score(peak_arr: np.ndarray, peak_range_arr: np.ndarray):
    z_score_arr = np.zeros(peak_arr.shape[0], dtype=np.float32)
    # for st, ed in peak_range_arr:
    for i in nb.prange(peak_range_arr.shape[0]):
        st, ed = peak_range_arr[i]
        if ed > st:
            ab = peak_arr[st:ed, 1]
            q1, q2, q3 = np.quantile(ab, q=[0.25, 0.5, 0.75])
            if q3 > q1:
                z_score_arr[st:ed] = (ab - q2) / (q3 - q1)
    return z_score_arr


@nb.njit(inline="always")
def find_peak_index(
    mz_arr: np.ndarray, theo_mz_arr: np.ndarray, tolerance_in_ppm: float
):
    mz_tol = theo_mz_arr * tolerance_in_ppm * 1e-6
    start_indices = np.searchsorted(mz_arr, theo_mz_arr - mz_tol, side="left").astype(
        np.uint32
    )
    stop_indices = np.searchsorted(mz_arr, theo_mz_arr + mz_tol, side="right").astype(
        np.uint32
    )

    return start_indices, stop_indices
