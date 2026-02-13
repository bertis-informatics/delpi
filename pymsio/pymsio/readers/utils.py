import math
import logging

import numpy as np
import numba as nb

logger = logging.getLogger(__name__)

try:
    import clr

    clr.AddReference("System")

    import ctypes

    from System.Runtime.InteropServices import GCHandle, GCHandleType
except Exception as e:
    logger.exception(
        "pythonnet/.NET runtime import failed. "
        f"Optional dependency unavailable: pythonnet/.NET import failed ({type(e).__name__}: {e}). "
        "Thermo RAW reader functionality will be disabled."
    )


def DotNetArrayToNPArray(src, dtype=np.float64):
    """
    See https://mail.python.org/pipermail/pythondotnet/2014-May/001527.html

    GCHandle pins the .NET array and np.frombuffer creates a zero-copy view.
    The pin is released in `finally`, so we must copy the data before returning;
    otherwise the numpy array becomes a dangling pointer and may segfault
    when the .NET GC compacts the heap.

    Pass dtype (e.g. np.float32) to convert in a single copy instead of two.
    """
    if src is None:
        return np.array([], dtype=dtype)
    src_hndl = GCHandle.Alloc(src, GCHandleType.Pinned)
    try:
        src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
        bufType = ctypes.c_double * len(src)
        cbuf = bufType.from_address(src_ptr)
        dest = np.array(cbuf, dtype=dtype)  # copy + convert while pinned
    finally:
        if src_hndl.IsAllocated:
            src_hndl.Free()
    return dest


def get_frame_num_to_index_arr(frame_nums):
    num_to_idx = np.zeros(frame_nums[-1] + 1, dtype=np.uint32)
    num_to_idx[frame_nums] = np.arange(len(frame_nums), dtype=np.uint32)
    return num_to_idx


@nb.njit(cache=True, fastmath=True)
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@nb.njit(parallel=True, fastmath=True, cache=True)
def compute_z_score_cdf_numba(
    ab_arr: np.ndarray, peak_range_arr: np.ndarray
) -> np.ndarray:
    """
    ab_arr: 1-D float32 intensity array of length N
    peak_range_arr: shape (M, 2) [start, end) integer
    returns: float32 array of length N in [0,1] (robust z -> normal CDF)
    """
    n = ab_arr.shape[0]
    out = np.empty(n, dtype=np.float32)

    for i in range(n):
        out[i] = 0.5

    for i in nb.prange(peak_range_arr.shape[0]):
        st = int(peak_range_arr[i, 0])
        ed = int(peak_range_arr[i, 1])
        if ed > st:
            ab = ab_arr[st:ed]
            # Numba(>=0.47)
            q1, q2, q3 = np.quantile(ab, np.array([0.25, 0.5, 0.75]))
            iqr = q3 - q1
            if iqr > 0.0:
                inv = 1.0 / iqr
                for j in range(st, ed):
                    z = (ab_arr[j] - q2) * inv
                    out[j] = _norm_cdf(z)

    return out
