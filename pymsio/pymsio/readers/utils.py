import math
import ctypes
from contextlib import contextmanager, ExitStack
from functools import lru_cache
from typing import Iterator, Tuple

import numpy as np
import numba as nb


@lru_cache(maxsize=1)
def _dotnet_handles():
    """Load optional .NET bindings only when a .NET array needs to be pinned."""
    try:
        import clr

        clr.AddReference("System")
        from System.Runtime.InteropServices import GCHandle, GCHandleType
    except Exception as error:
        raise ImportError(
            "Converting .NET arrays requires pythonnet and a working .NET runtime"
        ) from error
    return GCHandle, GCHandleType


def DotNetArrayToNPArray(src, dtype=np.float64):
    """Copy a .NET ``double[]`` into a new NumPy array of `dtype`.

    See https://mail.python.org/pipermail/pythondotnet/2014-May/001527.html.
    The data is copied while the array is pinned: a view would dangle once the
    pin is released and the .NET GC compacts the heap. Passing `dtype` (e.g.
    np.float32) converts in the same copy.
    """
    if src is None:
        return np.array([], dtype=dtype)
    GCHandle, GCHandleType = _dotnet_handles()
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


# NumPy dtype -> (.NET element type name, ctypes scalar type)
_DOTNET_ELEMENT_TYPES = {
    np.dtype(np.float64): ("Double", ctypes.c_double),
    np.dtype(np.float32): ("Single", ctypes.c_float),
}


def pin_dotnet_array(
    src, *, source_dtype: np.dtype = np.float64, validate: bool = True
):
    """Pin ``src`` and return ``(view, handle)``, a zero-copy read-only view.

    The caller must pass ``handle`` to :func:`unpin_dotnet_array` in a
    ``finally`` block once done with ``view``. Prefer the
    :func:`pinned_dotnet_array_view` context manager except in hot per-scan
    loops, where its generator overhead adds up. ``handle`` is ``None`` for a
    ``None`` or empty ``src``.
    """
    if src is None or len(src) == 0:
        return np.empty(0, dtype=source_dtype), None

    source_dtype = np.dtype(source_dtype)
    expected_name, ctypes_type = _DOTNET_ELEMENT_TYPES.get(source_dtype, (None, None))
    if ctypes_type is None:
        raise TypeError(f"Unsupported source_dtype for pinned view: {source_dtype!r}")

    GCHandle, GCHandleType = _dotnet_handles()
    if validate:
        element_type_name = src.GetType().GetElementType().Name
        if expected_name is not None and element_type_name != expected_name:
            raise TypeError(
                f"Unexpected .NET array element type {element_type_name!r}; "
                f"expected {expected_name!r} for source_dtype={source_dtype!r}"
            )

    n = len(src)
    src_hndl = GCHandle.Alloc(src, GCHandleType.Pinned)
    src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
    buf_type = ctypes_type * n
    cbuf = buf_type.from_address(src_ptr)
    view = np.frombuffer(cbuf, dtype=source_dtype)
    view.setflags(write=False)
    return view, src_hndl


def unpin_dotnet_array(handle) -> None:
    """Release a handle returned by :func:`pin_dotnet_array`. No-op for ``None``."""
    if handle is not None and handle.IsAllocated:
        handle.Free()


@contextmanager
def pinned_dotnet_array_view(
    src, *, source_dtype: np.dtype = np.float64, validate: bool = True
) -> Iterator[np.ndarray]:
    """Zero-copy, read-only NumPy view over a pinned .NET array.

    The handle is freed when the ``with`` block exits; the view must not be
    used after that, since the .NET GC may then move or collect the array.
    For hot per-scan loops, use :func:`pin_dotnet_array` and
    :func:`unpin_dotnet_array` directly.

    Args:
        src: a .NET array (e.g. ``System.Double[]``), or ``None``.
        source_dtype: NumPy dtype the .NET element type is checked against.
        validate: run that reflection check. It is a measurable pythonnet cost
            per scan, so callers that checked once per source can pass False.
    """
    view, handle = pin_dotnet_array(src, source_dtype=source_dtype, validate=validate)
    try:
        yield view
    finally:
        unpin_dotnet_array(handle)


@contextmanager
def pinned_dotnet_array_views(
    *arrays, source_dtype: np.dtype = np.float64, validate: bool = True
) -> Iterator[Tuple[np.ndarray, ...]]:
    """:func:`pinned_dotnet_array_view` for several arrays at once; every
    handle is released even if a later pin or the ``with`` body raises.
    """
    with ExitStack() as stack:
        views = tuple(
            stack.enter_context(
                pinned_dotnet_array_view(
                    arr, source_dtype=source_dtype, validate=validate
                )
            )
            for arr in arrays
        )
        yield views


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
    """Normal CDF of each peak's robust z-score within its spectrum.

    ``ab_arr`` is the (N,) float32 intensity array and ``peak_range_arr`` the
    (M, 2) integer ``[start, end)`` range of each spectrum. z is
    (ab - median) / IQR. Returns (N,) float32 in [0, 1]; 0.5 for peaks outside
    any range or in a spectrum with zero IQR.
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
            # np.quantile needs Numba >= 0.47
            q1, q2, q3 = np.quantile(ab, np.array([0.25, 0.5, 0.75]))
            iqr = q3 - q1
            if iqr > 0.0:
                inv = 1.0 / iqr
                for j in range(st, ed):
                    z = (ab_arr[j] - q2) * inv
                    out[j] = _norm_cdf(z)

    return out
