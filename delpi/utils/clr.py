import logging
import numpy as np

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

def DotNetArrayToNPArray(src):
    """
    See https://mail.python.org/pipermail/pythondotnet/2014-May/001527.html
    """
    if src is None:
        return np.array([], dtype=np.float64)
    src_hndl = GCHandle.Alloc(src, GCHandleType.Pinned)
    try:
        src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
        bufType = ctypes.c_double * len(src)
        cbuf = bufType.from_address(src_ptr)
        dest = np.frombuffer(cbuf, dtype=cbuf._type_).copy()
        # dest = np.frombuffer(cbuf, dtype=cbuf._type_)
    finally:
        if src_hndl.IsAllocated:
            src_hndl.Free()
        return dest  # noqa: B012