import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

from pymsio.readers.base import MassSpecFileReader
from pymsio.readers.ms_data import MassSpecData, PeakArray

logger = logging.getLogger(__name__)

ENV_DLL_DIR = "PYMSIO_THERMO_DLL_DIR"
REQUIRED_DLLS = [
    "ThermoFisher.CommonCore.Data.dll",
    "ThermoFisher.CommonCore.RawFileReader.dll",
]


def find_thermo_dll_dir() -> Path:
    candidates = []

    env = os.getenv(ENV_DLL_DIR)
    if env:
        candidates.append(Path(env))

    pkg_dir = Path(__file__).resolve().parents[1]
    pkg_path = pkg_dir / "dlls" / "thermo_fisher"
    cwd_path = Path.cwd() / "dlls" / "thermo_fisher"

    candidates.append(pkg_path)
    candidates.append(cwd_path)

    for d in candidates:
        if d and d.is_dir() and all((d / f).exists() for f in REQUIRED_DLLS):
            return d

    raise FileNotFoundError(
        "Thermo DLLs not found. Place the DLLs in one of the following locations:\n"
        f"- <set {ENV_DLL_DIR}>\n"
        f"- {pkg_path} (inside the installed pymsio package)\n"
        f"- {cwd_path} (relative to your working directory)\n"
        "Required:\n- " + "\n- ".join(REQUIRED_DLLS)
    )


LOADED_DLL = False
_DLL_LOAD_ERROR: str = ""

try:
    import clr

    clr.AddReference("System")
    import System

    from pymsio.readers.utils import pin_dotnet_array, unpin_dotnet_array

    dll_dir = find_thermo_dll_dir()

    for filename in REQUIRED_DLLS:
        clr.AddReference(os.path.join(dll_dir, filename))

    import ThermoFisher
    from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
    from ThermoFisher.CommonCore.Data.Interfaces import IScanEventBase

    LOADED_DLL = True
except Exception as exc:
    _DLL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"


class _PeakBufferBuilder:
    """Growable, contiguous ``float32`` (mz, ab) peak buffer with per-scan
    ``[start, stop)`` bookkeeping.

    Avoids building one small ``PeakArray`` per scan followed by a full-run
    ``np.concatenate()``: peaks are written directly into a preallocated
    buffer that grows geometrically (so most scans don't trigger a
    reallocation), and the final arrays are returned as plain slices/views
    (no trailing copy-to-shrink).
    """

    _GROWTH_FACTOR = 1.6
    _DEFAULT_INITIAL_CAPACITY = 1 << 16  # 65536 peaks

    def __init__(self, num_spectra: int, initial_capacity: Optional[int] = None):
        self.peak_start = np.empty(num_spectra, dtype=np.uint32)
        self.peak_stop = np.empty(num_spectra, dtype=np.uint32)
        self._capacity = max(initial_capacity or self._DEFAULT_INITIAL_CAPACITY, 1)
        self._mz = np.empty(self._capacity, dtype=np.float32)
        self._ab = np.empty(self._capacity, dtype=np.float32)
        self._used = 0
        # The .NET array element type (System.Double[]) doesn't change scan
        # to scan for a given RAW file/instrument, so validate it only once -
        # profiling showed the GetType().GetElementType().Name reflection
        # call is a real, measurable pythonnet boundary cost when repeated
        # per scan (~0.46s / 5000 scans in this env).
        self._type_validated = False

    def _ensure_capacity(self, extra: int) -> None:
        needed = self._used + extra
        if needed <= self._capacity:
            return
        new_capacity = max(needed, int(self._capacity * self._GROWTH_FACTOR) + 1)
        if new_capacity > np.iinfo(np.uint32).max:
            raise OverflowError(
                f"peak buffer would need {new_capacity} entries, "
                "exceeding the uint32 addressable range"
            )
        new_mz = np.empty(new_capacity, dtype=np.float32)
        new_ab = np.empty(new_capacity, dtype=np.float32)
        new_mz[: self._used] = self._mz[: self._used]
        new_ab[: self._used] = self._ab[: self._used]
        self._mz = new_mz
        self._ab = new_ab
        self._capacity = new_capacity

    def append_from_dotnet(self, masses, intensities, spectrum_index: int) -> None:
        """Filter (intensity > 0) and cast ``masses``/``intensities`` (.NET
        ``double[]``) directly into this builder's final buffer, while the
        arrays are pinned - no intermediate per-scan NumPy array is created.

        Uses the low-level ``pin_dotnet_array``/``unpin_dotnet_array`` pair
        directly (instead of the ``pinned_dotnet_array_view(s)`` context
        managers) since this runs once per scan: profiling showed the
        generator-based context manager protocol has measurable overhead of
        its own at that call frequency (tens of microseconds/call, adding up
        over tens of thousands of scans).
        """
        validate = not self._type_validated
        mz_view, mz_handle = pin_dotnet_array(masses, validate=validate)
        try:
            ab_view, ab_handle = pin_dotnet_array(intensities, validate=validate)
            try:
                self._type_validated = True
                n = min(mz_view.shape[0], ab_view.shape[0])
                st = self._used

                if n == 0:
                    self.peak_start[spectrum_index] = st
                    self.peak_stop[spectrum_index] = st
                    return

                sub_mz = mz_view[:n]
                sub_ab = ab_view[:n]

                if sub_ab.min() > 0:
                    # Fast path: every peak retained, no boolean mask needed.
                    self._ensure_capacity(n)
                    ed = self._used + n
                    self._mz[self._used : ed] = sub_mz
                    self._ab[self._used : ed] = sub_ab
                else:
                    mask = sub_ab > 0
                    count = int(np.count_nonzero(mask))
                    self._ensure_capacity(count)
                    ed = self._used + count
                    if count:
                        self._mz[self._used : ed] = sub_mz[mask]
                        self._ab[self._used : ed] = sub_ab[mask]

                self._used = ed
                self.peak_start[spectrum_index] = st
                self.peak_stop[spectrum_index] = ed
            finally:
                unpin_dotnet_array(ab_handle)
        finally:
            unpin_dotnet_array(mz_handle)

    def finish(self) -> Tuple[PeakArray, np.ndarray, np.ndarray]:
        """Return ``(peaks, peak_start, peak_stop)``. The peak arrays are
        contiguous views into the (possibly over-allocated) internal buffer -
        no copy is made to "shrink" them.
        """
        return (
            PeakArray(self._mz[: self._used], self._ab[: self._used]),
            self.peak_start,
            self.peak_stop,
        )


class ThermoRawReader(MassSpecFileReader):
    thread_safe = False

    def __init__(
        self,
        filepath: Union[str, Path],
        num_workers: int = 0,
    ):
        """Initialize the Thermo RAW reader.

        Note:
            ``num_workers`` is accepted for API compatibility but is
            currently unused: the underlying RAW handle is not thread-safe
            (see ``thread_safe = False``), so scans are read sequentially on
            the calling thread. Do not share ``self._raw`` across threads or
            processes.
        """
        if not LOADED_DLL:
            raise RuntimeError(f"Failed to load Thermo DLLs: {_DLL_LOAD_ERROR}")

        super().__init__(filepath, num_workers)

        self.filepath = str(filepath)
        self._raw_handle = RawFileReaderAdapter.FileFactory(self.filepath)
        self._raw_handle.SelectInstrument(
            ThermoFisher.CommonCore.Data.Business.Device.MS, 1
        )
        self._closed = False

        self._meta_df: Optional[pl.DataFrame] = None
        # Cached lightweight per-scan centroid flags, indexed by
        # ``frame_num - first_scan_number`` (populated by get_meta_df()/load()).
        self._is_centroid_by_index: Optional[np.ndarray] = None
        # Validate the .NET array element type only once for get_frame()/
        # get_frames() (see _PeakBufferBuilder._type_validated for why).
        self._peak_view_validated = False

        # Optional call-count instrumentation for benchmarking/profiling
        # only; ``None`` (disabled) has negligible overhead in the hot loop.
        self._debug_call_counts: Optional[Dict[str, int]] = None
        self._warning_counts: Dict[str, int] = {}

    @property
    def _raw(self):
        """The underlying pythonnet RAW handle.

        Routed through a property so every access point raises a clear
        error once the reader has been closed, instead of failing deep
        inside a .NET call with an opaque error.
        """
        if self._closed or self._raw_handle is None:
            raise RuntimeError("ThermoRawReader is closed")
        return self._raw_handle

    def close(self):
        """Idempotent: safe to call multiple times."""
        if self._closed:
            return
        if self._raw_handle is not None:
            self._raw_handle.Dispose()
            self._raw_handle = None
        self._closed = True
        self._is_centroid_by_index = None

    def enable_call_counting(self) -> Dict[str, int]:
        """Benchmarking/profiling helper: start counting calls to the
        underlying Thermo API methods. Returns the live counts dict."""
        self._debug_call_counts = defaultdict(int)
        return self._debug_call_counts

    def disable_call_counting(self) -> None:
        self._debug_call_counts = None

    @property
    def acquisition_date(self) -> str:
        return self._raw.CreationDate.ToString("o")

    @property
    def num_spectra(self):
        return self.num_frames

    @property
    def num_frames(self) -> int:
        return (
            self._raw.RunHeaderEx.LastSpectrum - self._raw.RunHeaderEx.FirstSpectrum + 1
        )

    @property
    def first_scan_number(self) -> int:
        return self._raw.RunHeaderEx.FirstSpectrum

    @property
    def last_scan_number(self) -> int:
        return self._raw.RunHeaderEx.LastSpectrum

    @property
    def instrument(self) -> str:
        return System.String.Join(
            " -> ", self._raw.GetAllInstrumentNamesFromInstrumentMethod()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warn(self, key: str) -> None:
        self._warning_counts[key] = self._warning_counts.get(key, 0) + 1

    def _emit_warnings(self) -> None:
        if not self._warning_counts:
            return
        for key, count in self._warning_counts.items():
            logger.warning(
                "ThermoRawReader(%s): %d scan(s) affected by: %s",
                self.filepath,
                count,
                key,
            )
        self._warning_counts.clear()

    def _resolve_isolation(self, scan_event, ms_level: int):
        """Returns ``(isolation_min_mz, isolation_max_mz)``, both ``None``
        for MS1 or when isolation metadata can't be determined."""
        if ms_level == 1:
            return None, None

        counts = self._debug_call_counts
        if counts is not None:
            counts["GetReaction"] += 1

        try:
            reaction = scan_event.GetReaction(0)
        except Exception:
            self._warn("reaction_unavailable")
            return None, None

        try:
            if reaction.PrecursorRangeIsValid:
                return float(reaction.FirstPrecursorMass), float(
                    reaction.LastPrecursorMass
                )
            center = float(reaction.PrecursorMass)
            width = float(reaction.IsolationWidth)
            lo = center - width / 2.0
            return lo, lo + width
        except Exception:
            self._warn("isolation_metadata_invalid")
            return None, None

    def _peak_array_from_dotnet(self, masses, intensities) -> PeakArray:
        """Filter (intensity > 0) + cast to float32 a single scan's
        masses/intensities into an independent :class:`PeakArray`, without
        an intermediate per-scan NumPy array (values are copied directly
        out of the pinned .NET buffers).
        """
        validate = not self._peak_view_validated
        mz_view, mz_handle = pin_dotnet_array(masses, validate=validate)
        try:
            ab_view, ab_handle = pin_dotnet_array(intensities, validate=validate)
            try:
                self._peak_view_validated = True
                n = min(mz_view.shape[0], ab_view.shape[0])
                if n == 0:
                    return PeakArray.empty()

                sub_mz = mz_view[:n]
                sub_ab = ab_view[:n]

                if sub_ab.min() > 0:
                    return PeakArray(
                        sub_mz.astype(np.float32), sub_ab.astype(np.float32)
                    )

                mask = sub_ab > 0
                if not mask.any():
                    return PeakArray.empty()
                return PeakArray(
                    sub_mz[mask].astype(np.float32), sub_ab[mask].astype(np.float32)
                )
            finally:
                unpin_dotnet_array(ab_handle)
        finally:
            unpin_dotnet_array(mz_handle)

    def _cached_centroid(self, frame_num: int) -> bool:
        cache = self._is_centroid_by_index
        if cache is not None:
            idx = frame_num - self.first_scan_number
            if 0 <= idx < len(cache):
                return bool(cache[idx])

        counts = self._debug_call_counts
        if counts is not None:
            counts["IsCentroidScanFromScanNumber"] += 1
        return bool(self._raw.IsCentroidScanFromScanNumber(frame_num))

    # ------------------------------------------------------------------
    # Metadata-only path
    # ------------------------------------------------------------------

    def get_meta_df(self) -> pl.DataFrame:
        """Read metadata for all scans without retrieving any peak data."""
        if self._meta_df is not None:
            return self._meta_df

        n = self.num_frames
        first = self.first_scan_number

        frame_num = np.arange(first, first + n, dtype=np.uint32)
        mz_lo = np.empty(n, dtype=np.float32)
        mz_hi = np.empty(n, dtype=np.float32)
        time_in_seconds = np.empty(n, dtype=np.float32)
        ms_level = np.empty(n, dtype=np.uint8)
        isolation_min_mz = np.full(n, np.nan, dtype=np.float32)
        isolation_max_mz = np.full(n, np.nan, dtype=np.float32)
        is_centroid = np.empty(n, dtype=np.bool_)

        raw = self._raw
        get_stats = raw.GetScanStatsForScanNumber
        get_event = raw.GetScanEventForScanNumber
        counts = self._debug_call_counts

        for i in self._progress(range(n), desc="read meta"):
            fn = first + i
            if counts is not None:
                counts["GetScanStatsForScanNumber"] += 1
                counts["GetScanEventForScanNumber"] += 1

            stats = get_stats(fn)
            event = IScanEventBase(get_event(fn))

            try:
                rt = float(stats.StartTime)  # minutes
            except AttributeError:
                rt = float(raw.RetentionTimeFromScanNumber(fn))

            time_in_seconds[i] = rt * 60.0
            mz_lo[i] = float(stats.LowMass)
            mz_hi[i] = float(stats.HighMass)
            is_centroid[i] = bool(stats.IsCentroidScan)

            lvl = int(event.MSOrder)
            ms_level[i] = lvl

            lo, hi = self._resolve_isolation(event, lvl)
            if lo is not None:
                isolation_min_mz[i] = lo
                isolation_max_mz[i] = hi

        self._is_centroid_by_index = is_centroid
        self._emit_warnings()

        meta_df = pl.DataFrame(
            {
                "frame_num": frame_num,
                "mz_lo": mz_lo,
                "mz_hi": mz_hi,
                "time_in_seconds": time_in_seconds,
                "ms_level": ms_level,
                "isolation_min_mz": isolation_min_mz,
                "isolation_max_mz": isolation_max_mz,
            },
            schema=self.meta_schema,
            nan_to_null=True,
        )
        self._meta_df = meta_df
        return meta_df

    # ------------------------------------------------------------------
    # Single-scan access
    # ------------------------------------------------------------------

    def get_frame(self, frame_num: int) -> PeakArray:
        centroid = self._cached_centroid(frame_num)
        counts = self._debug_call_counts

        if not centroid:
            if counts is not None:
                counts["GetSimplifiedCentroids"] += 1
            data = self._raw.GetSimplifiedCentroids(frame_num)
        else:
            if counts is not None:
                counts["GetSimplifiedScan"] += 1
            data = self._raw.GetSimplifiedScan(frame_num)

        return self._peak_array_from_dotnet(data.Masses, data.Intensities)

    def get_frames(self, frame_nums: Sequence[int]) -> List[PeakArray]:
        """Preserves the requested order and duplicate frame numbers."""
        return [
            self.get_frame(int(fn))
            for fn in self._progress(frame_nums, desc="load spectra")
        ]

    # ------------------------------------------------------------------
    # Bulk load
    # ------------------------------------------------------------------

    def load(self, progress=None) -> MassSpecData:
        need_meta = self._meta_df is None

        n = self.num_frames
        first = self.first_scan_number
        builder = _PeakBufferBuilder(n)

        raw = self._raw
        get_scan = raw.GetSimplifiedScan
        get_centroids = raw.GetSimplifiedCentroids
        counts = self._debug_call_counts

        if need_meta:
            frame_num = np.arange(first, first + n, dtype=np.uint32)
            mz_lo = np.empty(n, dtype=np.float32)
            mz_hi = np.empty(n, dtype=np.float32)
            time_in_seconds = np.empty(n, dtype=np.float32)
            ms_level = np.empty(n, dtype=np.uint8)
            isolation_min_mz = np.full(n, np.nan, dtype=np.float32)
            isolation_max_mz = np.full(n, np.nan, dtype=np.float32)
            is_centroid = np.empty(n, dtype=np.bool_)

            get_stats = raw.GetScanStatsForScanNumber
            get_event = raw.GetScanEventForScanNumber

            for i in self._progress(range(n), progress=progress, desc="load spectra"):
                fn = first + i
                if counts is not None:
                    counts["GetScanStatsForScanNumber"] += 1
                    counts["GetScanEventForScanNumber"] += 1

                stats = get_stats(fn)
                event = IScanEventBase(get_event(fn))

                try:
                    rt = float(stats.StartTime)
                except AttributeError:
                    rt = float(raw.RetentionTimeFromScanNumber(fn))

                time_in_seconds[i] = rt * 60.0
                mz_lo[i] = float(stats.LowMass)
                mz_hi[i] = float(stats.HighMass)
                centroid = bool(stats.IsCentroidScan)
                is_centroid[i] = centroid

                lvl = int(event.MSOrder)
                ms_level[i] = lvl
                lo, hi = self._resolve_isolation(event, lvl)
                if lo is not None:
                    isolation_min_mz[i] = lo
                    isolation_max_mz[i] = hi

                if not centroid:
                    if counts is not None:
                        counts["GetSimplifiedCentroids"] += 1
                    data = get_centroids(fn)
                else:
                    if counts is not None:
                        counts["GetSimplifiedScan"] += 1
                    data = get_scan(fn)
                builder.append_from_dotnet(data.Masses, data.Intensities, i)

            self._is_centroid_by_index = is_centroid
            self._emit_warnings()

            self._meta_df = pl.DataFrame(
                {
                    "frame_num": frame_num,
                    "mz_lo": mz_lo,
                    "mz_hi": mz_hi,
                    "time_in_seconds": time_in_seconds,
                    "ms_level": ms_level,
                    "isolation_min_mz": isolation_min_mz,
                    "isolation_max_mz": isolation_max_mz,
                },
                schema=self.meta_schema,
                nan_to_null=True,
            )
        else:
            # Metadata already cached: reuse cached centroid flags and skip
            # scan-stats/scan-event/reaction calls entirely.
            centroid_flags = self._is_centroid_by_index
            is_centroid_available = centroid_flags is not None

            for i in self._progress(range(n), progress=progress, desc="load spectra"):
                fn = first + i
                if is_centroid_available:
                    centroid = bool(centroid_flags[i])
                else:
                    centroid = self._cached_centroid(fn)

                if not centroid:
                    if counts is not None:
                        counts["GetSimplifiedCentroids"] += 1
                    data = get_centroids(fn)
                else:
                    if counts is not None:
                        counts["GetSimplifiedScan"] += 1
                    data = get_scan(fn)
                builder.append_from_dotnet(data.Masses, data.Intensities, i)

        peaks, peak_start, peak_stop = builder.finish()

        return MassSpecData.create_from_flat(
            run_name=self.run_name,
            meta_df=self._meta_df,
            peaks=peaks,
            peak_start=peak_start,
            peak_stop=peak_stop,
        )
