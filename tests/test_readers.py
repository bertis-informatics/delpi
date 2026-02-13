import numpy as np
import polars as pl

from pymsio.readers.ms_data import PeakArray


# ---------------------------------------------------------------------------
# Basic import / environment tests (always run, no file needed)
# ---------------------------------------------------------------------------


class TestImports:

    def test_pymsio_import(self):
        import pymsio.readers

        assert hasattr(pymsio.readers, "ReaderFactory")

    def test_base_module(self):
        from pymsio.readers.base import MassSpecFileReader
        from pymsio.readers.ms_data import MassSpecData

        assert MassSpecFileReader is not None
        assert MassSpecData is not None

    def test_mzml_import(self):
        from pymsio.readers.mzml import MzmlFileReader

        assert MzmlFileReader is not None

    def test_thermo_import(self):
        from pymsio.readers.thermo import ThermoRawReader

        assert ThermoRawReader is not None

    def test_thermo_dll_loaded(self):
        from pymsio.readers.thermo import LOADED_DLL

        assert LOADED_DLL, (
            "Thermo DLLs not loaded. "
            "Ensure DLLs are in pymsio/dlls/thermo_fisher/ or PYMSIO_THERMO_DLL_DIR is set."
        )

    def test_reader_factory_supported_extensions(self):
        from pymsio.readers import ReaderFactory

        assert ".raw" in ReaderFactory.supported_file_extensions
        assert ".mzml" in ReaderFactory.supported_file_extensions


def _validate_meta_df(meta_df: pl.DataFrame) -> None:
    """Common assertions for any reader's meta DataFrame."""
    assert isinstance(meta_df, pl.DataFrame)
    assert meta_df.shape[0] > 0, "meta_df should not be empty"

    for col in ("frame_num", "time_in_seconds", "ms_level", "mz_lo", "mz_hi"):
        assert col in meta_df.columns, f"missing column: {col}"

    assert meta_df["frame_num"].is_sorted(), "frame_num should be sorted"
    assert (meta_df["ms_level"] >= 1).all(), "ms_level should be >= 1"


def _validate_peaks(peaks: PeakArray) -> None:
    """Common assertions for a single frame's peak array."""
    assert isinstance(peaks, PeakArray)
    assert peaks.mz.dtype == np.float32
    assert peaks.ab.dtype == np.float32
    assert peaks.mz.ndim == 1
    assert peaks.ab.ndim == 1
    assert len(peaks.mz) == len(peaks.ab)


def _validate_mass_spec_data(ms_data) -> None:
    """Common assertions for MassSpecData returned by load()."""
    assert ms_data is not None
    assert ms_data.meta_df.shape[0] > 0
    assert isinstance(ms_data.peaks, PeakArray)
    assert ms_data.peaks.mz.ndim == 1
    assert ms_data.peaks.ab.ndim == 1
    assert ms_data.peaks.mz.dtype == np.float32
    assert ms_data.peaks.ab.dtype == np.float32
    assert len(ms_data.peaks) == len(ms_data.peaks.mz)


# ---------------------------------------------------------------------------
# MzML reader tests
# ---------------------------------------------------------------------------


class TestMzmlReader:

    def test_get_meta_df(self, mzml_path):
        from pymsio.readers.mzml import MzmlFileReader

        reader = MzmlFileReader(mzml_path)
        meta_df = reader.get_meta_df()
        _validate_meta_df(meta_df)

    def test_get_frame(self, mzml_path):
        from pymsio.readers.mzml import MzmlFileReader

        reader = MzmlFileReader(mzml_path)
        meta_df = reader.get_meta_df()
        first_frame = int(meta_df["frame_num"][0])
        peaks = reader.get_frame(first_frame)
        _validate_peaks(peaks)

    def test_load(self, mzml_path):
        from pymsio.readers.mzml import MzmlFileReader

        reader = MzmlFileReader(mzml_path)
        ms_data = reader.load()
        _validate_mass_spec_data(ms_data)


# ---------------------------------------------------------------------------
# Thermo RAW reader tests
# ---------------------------------------------------------------------------


class TestThermoReader:

    def test_get_meta_df(self, raw_path):
        from pymsio.readers.thermo import ThermoRawReader

        reader = ThermoRawReader(raw_path)
        meta_df = reader.get_meta_df()
        _validate_meta_df(meta_df)
        reader.close()

    def test_get_frame(self, raw_path):
        from pymsio.readers.thermo import ThermoRawReader

        reader = ThermoRawReader(raw_path)
        first_frame = reader.first_scan_number
        peaks = reader.get_frame(first_frame)
        _validate_peaks(peaks)
        reader.close()

    def test_load(self, raw_path):
        from pymsio.readers.thermo import ThermoRawReader

        reader = ThermoRawReader(raw_path)
        ms_data = reader.load()
        _validate_mass_spec_data(ms_data)
        reader.close()
