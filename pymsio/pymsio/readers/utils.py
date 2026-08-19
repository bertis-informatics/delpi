import math
import logging
from contextlib import contextmanager, ExitStack
from typing import Iterator, Tuple

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


# Maps a NumPy source dtype to (the expected .NET array element type name,
# the equivalent ctypes scalar type) used by `pinned_dotnet_array_view`.
try:
    _DOTNET_ELEMENT_TYPES = {
        np.dtype(np.float64): ("Double", ctypes.c_double),
        np.dtype(np.float32): ("Single", ctypes.c_float),
    }
except NameError:
    # pythonnet/.NET unavailable (see import guard above); Thermo-specific
    # helpers below simply won't be usable in that case.
    _DOTNET_ELEMENT_TYPES = {}


def pin_dotnet_array(
    src, *, source_dtype: np.dtype = np.float64, validate: bool = True
):
    """Low-level primitive behind :func:`pinned_dotnet_array_view`.

    Pins ``src`` and returns ``(view, handle)``. The caller **must** call
    :func:`unpin_dotnet_array(handle)` in a ``finally`` block once done with
    ``view`` - prefer the :func:`pinned_dotnet_array_view` context manager
    unless profiling shows its overhead matters (e.g. many thousands of
    calls in a hot per-scan loop): a context manager built on a generator
    function has measurable overhead of its own (protocol dispatch, the
    generator's extra ``next()`` call, etc.) that is worth avoiding in a
    tight loop called once per scan.

    ``handle`` is ``None`` when ``src`` is ``None``/empty (nothing to pin);
    :func:`unpin_dotnet_array` is a no-op for ``None``.
    """
    if src is None or len(src) == 0:
        return np.empty(0, dtype=source_dtype), None

    source_dtype = np.dtype(source_dtype)
    expected_name, ctypes_type = _DOTNET_ELEMENT_TYPES.get(source_dtype, (None, None))
    if ctypes_type is None:
        raise TypeError(f"Unsupported source_dtype for pinned view: {source_dtype!r}")

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
    """Temporarily expose a zero-copy, read-only NumPy view over a pinned
    .NET array.

    Unlike :func:`DotNetArrayToNPArray`, this does **not** copy: the yielded
    view aliases the pinned .NET memory directly. The GCHandle is freed as
    soon as the ``with`` block exits, so the view MUST NOT be used (read,
    stored, or returned) after the context manager exits - doing so is
    equivalent to holding a dangling pointer once the .NET GC can move or
    collect the underlying array.

    Args:
        src: a .NET array (e.g. ``System.Double[]``), or ``None``.
        source_dtype: expected NumPy-equivalent element dtype. The actual
            .NET element type is validated against this before pinning.
        validate: if ``False``, skip the ``GetType().GetElementType().Name``
            reflection check. This check is a real (measured) pythonnet
            boundary cost when repeated per-scan in a hot loop; callers that
            already validated the element type once for a given array source
            (e.g. once per RAW file/instrument) should pass ``False`` on
            subsequent calls.

    Note:
        For very hot loops (thousands of calls, e.g. once per scan), prefer
        the lower-level :func:`pin_dotnet_array`/:func:`unpin_dotnet_array`
        pair with an explicit ``try``/``finally`` - this context manager's
        generator-based protocol has measurable overhead of its own at that
        call frequency.
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
    """Like :func:`pinned_dotnet_array_view`, but pins/yields multiple .NET
    arrays at once (e.g. masses + intensities), guaranteeing every GCHandle
    is released even if pinning or use of a later array raises.
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
