from typing import List

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
# Focused regression tests for the mzML reader optimization
# (metadata-only parsing must not decode peaks; load() must be single-pass;
# missing/empty binary arrays must be handled safely)
# ---------------------------------------------------------------------------

_MINIMAL_MZML_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml">
  <run id="run1">
    <spectrumList count="{count}">
      {spectra}
    </spectrumList>
  </run>
</mzML>
"""

_FULL_SPECTRUM = """
      <spectrum id="scan={index}" index="{index}" defaultArrayLength="{n}">
        <cvParam accession="MS:1000511" name="ms level" value="{ms_level}"/>
        <scanList count="1">
          <scan>
            <cvParam accession="MS:1000016" name="scan start time" value="{rt}" unitAccession="UO:0000031" unitName="minute"/>
            <scanWindowList count="1">
              <scanWindow>
                <cvParam accession="MS:1000501" name="scan window lower limit" value="200.0"/>
                <cvParam accession="MS:1000500" name="scan window upper limit" value="1500.0"/>
              </scanWindow>
            </scanWindowList>
          </scan>
        </scanList>
        <binaryDataArrayList count="2">
          <binaryDataArray encodedLength="{mz_len}">
            <cvParam accession="MS:1000521" name="32-bit float"/>
            <cvParam accession="MS:1000576" name="no compression"/>
            <cvParam accession="MS:1000514" name="m/z array"/>
            <binary>{mz_b64}</binary>
          </binaryDataArray>
          <binaryDataArray encodedLength="{ab_len}">
            <cvParam accession="MS:1000521" name="32-bit float"/>
            <cvParam accession="MS:1000576" name="no compression"/>
            <cvParam accession="MS:1000515" name="intensity array"/>
            <binary>{ab_b64}</binary>
          </binaryDataArray>
        </binaryDataArrayList>
      </spectrum>
"""

_SPECTRUM_NO_ARRAYS = """
      <spectrum id="scan={index}" index="{index}" defaultArrayLength="0">
        <cvParam accession="MS:1000511" name="ms level" value="1"/>
        <scanList count="1">
          <scan>
            <cvParam accession="MS:1000016" name="scan start time" value="0.1" unitAccession="UO:0000031" unitName="minute"/>
          </scan>
        </scanList>
      </spectrum>
"""


def _make_mzml_with_spectra(tmp_path, spectra_xml: List[str]) -> str:
    content = _MINIMAL_MZML_TEMPLATE.format(
        count=len(spectra_xml), spectra="".join(spectra_xml)
    )
    path = tmp_path / "synthetic.mzML"
    path.write_text(content, encoding="utf-8")
    return str(path)


def _make_full_spectrum(index: int, mz, ab, ms_level=2, rt=1.5) -> str:
    import base64
    import numpy as np

    mz_arr = np.asarray(mz, dtype=np.float32)
    ab_arr = np.asarray(ab, dtype=np.float32)
    mz_b64 = base64.b64encode(mz_arr.tobytes()).decode("ascii")
    ab_b64 = base64.b64encode(ab_arr.tobytes()).decode("ascii")
    return _FULL_SPECTRUM.format(
        index=index,
        n=len(mz_arr),
        ms_level=ms_level,
        rt=rt,
        mz_len=len(mz_b64),
        ab_len=len(ab_b64),
        mz_b64=mz_b64,
        ab_b64=ab_b64,
    )


class TestMzmlReaderOptimizations:
    def test_get_meta_df_does_not_decode_peaks(self, tmp_path, monkeypatch):
        """Metadata-only parsing must not call the binary decoder at all."""
        from pymsio.readers import mzml as mzml_mod

        spec = _make_full_spectrum(0, [100.0, 200.0], [10.0, 20.0])
        path = _make_mzml_with_spectra(tmp_path, [spec])

        def _boom(*args, **kwargs):
            raise AssertionError(
                "binary_decode should not be called for meta-only parsing"
            )

        monkeypatch.setattr(mzml_mod, "binary_decode", _boom)

        reader = mzml_mod.MzmlFileReader(path)
        meta_df = reader.get_meta_df()
        assert meta_df.shape[0] == 1
        assert meta_df["ms_level"][0] == 2

    def test_load_is_single_pass_when_meta_not_cached(self, tmp_path, monkeypatch):
        """load() must open/parse the file exactly once when metadata isn't cached yet."""
        from pymsio.readers import mzml as mzml_mod

        specs = [_make_full_spectrum(i, [100.0 + i], [10.0]) for i in range(3)]
        path = _make_mzml_with_spectra(tmp_path, specs)

        reader = mzml_mod.MzmlFileReader(path)
        open_calls = []
        orig_open = reader._open_file_handle

        def _counting_open():
            open_calls.append(1)
            return orig_open()

        monkeypatch.setattr(reader, "_open_file_handle", _counting_open)

        ms_data = reader.load()
        assert len(open_calls) == 1
        assert ms_data.meta_df.shape[0] == 3
        assert len(ms_data.peaks.mz) == 3

    def test_get_meta_df_then_load_remains_valid(self, tmp_path):
        from pymsio.readers.mzml import MzmlFileReader

        specs = [
            _make_full_spectrum(i, [100.0 + i, 300.0], [10.0, 5.0]) for i in range(2)
        ]
        path = _make_mzml_with_spectra(tmp_path, specs)

        reader = MzmlFileReader(path)
        meta_df = reader.get_meta_df()
        ms_data = reader.load()

        assert meta_df.shape[0] == 2
        assert ms_data.meta_df.shape[0] == 2
        assert len(ms_data.peaks.mz) == 4

    def test_missing_binary_arrays_handled_safely(self, tmp_path):
        """A spectrum with no <binaryDataArrayList> must not crash parsing."""
        from pymsio.readers.mzml import MzmlFileReader

        specs = [_SPECTRUM_NO_ARRAYS.format(index=0)]
        path = _make_mzml_with_spectra(tmp_path, specs)

        reader = MzmlFileReader(path)
        meta_df = reader.get_meta_df()
        assert meta_df.shape[0] == 1

        ms_data = reader.load()
        assert ms_data.meta_df.shape[0] == 1
        assert len(ms_data.peaks.mz) == 0

    def test_non_positive_intensity_peaks_filtered(self, tmp_path):
        from pymsio.readers.mzml import MzmlFileReader

        spec = _make_full_spectrum(0, [100.0, 200.0, 300.0], [10.0, 0.0, -5.0])
        path = _make_mzml_with_spectra(tmp_path, [spec])

        reader = MzmlFileReader(path)
        ms_data = reader.load()
        assert len(ms_data.peaks.mz) == 1
        assert ms_data.peaks.mz[0] == 100.0
        assert ms_data.peaks.mz.dtype == np.float32
        assert ms_data.peaks.ab.dtype == np.float32


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


# ---------------------------------------------------------------------------
# Focused regression tests for the Thermo RAW reader optimization
# ---------------------------------------------------------------------------


class TestThermoReaderOptimizations:
    def test_close_is_idempotent_and_blocks_further_use(self, raw_path):
        from pymsio.readers.thermo import ThermoRawReader

        reader = ThermoRawReader(raw_path)
        reader.close()
        reader.close()  # must not raise

        import pytest

        with pytest.raises(RuntimeError):
            reader.get_meta_df()

    def test_get_meta_df_then_load_matches_direct_load(self, raw_path):
        from pymsio.readers.thermo import ThermoRawReader

        r1 = ThermoRawReader(raw_path)
        direct_data = r1.load()
        r1.close()

        r2 = ThermoRawReader(raw_path)
        meta_df = r2.get_meta_df()
        two_pass_data = r2.load()
        r2.close()

        # get_meta_df() only returns the metadata columns (no peak_start/
        # peak_stop, which load() adds), so compare on the common columns.
        assert meta_df.equals(direct_data.meta_df.select(meta_df.columns))
        assert two_pass_data.meta_df.equals(direct_data.meta_df)
        assert np.array_equal(direct_data.peaks.mz, two_pass_data.peaks.mz)
        assert np.array_equal(direct_data.peaks.ab, two_pass_data.peaks.ab)

    def test_load_reuses_cached_centroid_flags_and_skips_meta_calls(self, raw_path):
        """After get_meta_df(), a subsequent load() must not repeat
        GetScanStatsForScanNumber / GetScanEventForScanNumber / GetReaction,
        and must not call IsCentroidScanFromScanNumber (cached flags used
        instead)."""
        from pymsio.readers.thermo import ThermoRawReader

        reader = ThermoRawReader(raw_path)
        reader.get_meta_df()
        n = reader.num_frames

        counts = reader.enable_call_counting()
        reader.load()
        reader.disable_call_counting()
        reader.close()

        assert counts.get("GetScanStatsForScanNumber", 0) == 0
        assert counts.get("GetScanEventForScanNumber", 0) == 0
        assert counts.get("GetReaction", 0) == 0
        assert counts.get("IsCentroidScanFromScanNumber", 0) == 0
        # One spectrum-retrieval call per scan (split across the two APIs).
        assert (
            counts.get("GetSimplifiedScan", 0) + counts.get("GetSimplifiedCentroids", 0)
            == n
        )

    def test_direct_load_calls_scan_stats_and_event_once_per_scan(self, raw_path):
        from pymsio.readers.thermo import ThermoRawReader

        reader = ThermoRawReader(raw_path)
        counts = reader.enable_call_counting()
        reader.load()
        n = reader.num_frames
        reader.close()

        assert counts.get("GetScanStatsForScanNumber", 0) == n
        assert counts.get("GetScanEventForScanNumber", 0) == n
        assert counts.get("IsCentroidScanFromScanNumber", 0) == 0

    def test_get_frames_preserves_order_and_duplicates(self, raw_path):
        from pymsio.readers.thermo import ThermoRawReader

        reader = ThermoRawReader(raw_path)
        first = reader.first_scan_number
        requested = [first + 2, first, first, first + 1]
        results = reader.get_frames(requested)
        reader.close()

        assert len(results) == len(requested)
        for peaks in results:
            _validate_peaks(peaks)
        # duplicate frame numbers must yield equal (independent) PeakArrays
        assert np.array_equal(results[1].mz, results[2].mz)
        assert np.array_equal(results[1].ab, results[2].ab)

    def test_peak_start_stop_semantics(self, raw_path):
        from pymsio.readers.thermo import ThermoRawReader

        reader = ThermoRawReader(raw_path)
        ms_data = reader.load()
        reader.close()

        peak_start = ms_data.meta_df["peak_start"].to_numpy()
        peak_stop = ms_data.meta_df["peak_stop"].to_numpy()
        assert (peak_stop >= peak_start).all()
        assert (peak_start[1:] >= peak_stop[:-1]).all()
        assert peak_stop[-1] == len(ms_data.peaks.mz)
