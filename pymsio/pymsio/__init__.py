from .readers.factory import ReaderFactory
from .readers.ms_data import MassSpecData
from .readers.base import MassSpecFileReader
from .readers.mzml import MzmlFileReader

# ThermoRawReader is imported on first access (it loads pythonnet/.NET), so it
# is left out of __all__: `from pymsio import *` must not start the runtime.
__all__ = [
    "ReaderFactory",
    "MassSpecData",
    "MassSpecFileReader",
    "MzmlFileReader",
]

_LAZY_ATTRS = ("ThermoRawReader",)


def __getattr__(name):
    if name == "ThermoRawReader":
        from .readers.thermo import ThermoRawReader

        globals()[name] = ThermoRawReader
        return ThermoRawReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
