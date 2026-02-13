import os
from pathlib import Path

import numpy as np
import polars as pl
from pathlib import Path
from collections import defaultdict
from typing import Sequence, Union, List, Optional, Dict, Any

from pymsio.readers.base import MassSpecFileReader
from pymsio.readers.ms_data import MassSpecData, PeakArray

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

    from pymsio.readers.utils import DotNetArrayToNPArray

    dll_dir = find_thermo_dll_dir()

    for filename in REQUIRED_DLLS:
        clr.AddReference(os.path.join(dll_dir, filename))

    import ThermoFisher
    from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
    from ThermoFisher.CommonCore.Data.Interfaces import IScanEvent, IScanEventBase

    LOADED_DLL = True
except Exception as exc:
    _DLL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"


class ThermoRawReader(MassSpecFileReader):
    thread_safe = False

    def __init__(
        self,
        filepath: Union[str, Path],
        num_workers: int = 0,
        show_progress: bool = False,
    ):
        if not LOADED_DLL:
            raise RuntimeError(f"Failed to load Thermo DLLs: {_DLL_LOAD_ERROR}")

        super().__init__(filepath, num_workers, show_progress)

        self.filepath = str(filepath)
        self._raw = RawFileReaderAdapter.FileFactory(self.filepath)
        self._raw.SelectInstrument(ThermoFisher.CommonCore.Data.Business.Device.MS, 1)

        self._meta_df: Optional[pl.DataFrame] = None

    def close(self):
        if self._raw is not None:
            self._raw.Dispose()
            self._raw = None

    @property
    def acquisition_date(self) -> str:
        return self._raw.CreationDate.ToString("o")

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

    def _read_peaks_arrays(self, frame_num: int) -> PeakArray:
        is_centroid = self._raw.IsCentroidScanFromScanNumber(frame_num)

        if not is_centroid:
            data = self._raw.GetSimplifiedCentroids(frame_num)
        else:
            data = self._raw.GetSimplifiedScan(frame_num)

        mz_arr = DotNetArrayToNPArray(data.Masses, dtype=np.float32)
        inten_arr = DotNetArrayToNPArray(data.Intensities, dtype=np.float32)

        if mz_arr is None or len(mz_arr) == 0:
            return PeakArray.empty()

        mask = inten_arr > 0
        n = np.count_nonzero(mask)

        if n == 0:
            return PeakArray.empty()

        if n == len(mz_arr):
            return PeakArray(mz_arr, inten_arr)

        idx = np.flatnonzero(mask)
        return PeakArray(mz_arr[idx], inten_arr[idx])

    def _read_scan_meta(self, frame_num: int, cols: Dict[str, list]) -> None:
        scan_stats = self._raw.GetScanStatsForScanNumber(frame_num)
        scan_event = IScanEventBase(self._raw.GetScanEventForScanNumber(frame_num))

        try:
            rt = float(scan_stats.StartTime)  # minutes
        except AttributeError:
            rt = float(self._raw.RetentionTimeFromScanNumber(frame_num))

        ms_level = int(scan_event.MSOrder)

        if ms_level == 1:
            isolation_min_mz = None
            isolation_max_mz = None
        else:
            reaction = scan_event.GetReaction(0)
            if reaction.PrecursorRangeIsValid:
                isolation_min_mz = reaction.FirstPrecursorMass
                isolation_max_mz = reaction.LastPrecursorMass
            else:
                isolation_center = float(reaction.PrecursorMass)
                isolation_width = float(reaction.IsolationWidth)
                isolation_min_mz = isolation_center - isolation_width / 2.0
                isolation_max_mz = isolation_min_mz + isolation_width

        cols["frame_num"].append(frame_num)
        cols["time_in_seconds"].append(rt * 60)
        cols["ms_level"].append(ms_level)
        cols["isolation_min_mz"].append(isolation_min_mz)
        cols["isolation_max_mz"].append(isolation_max_mz)
        cols["mz_lo"].append(float(scan_stats.LowMass))
        cols["mz_hi"].append(float(scan_stats.HighMass))

    def _build_meta_df(self, cols: Dict[str, list]) -> pl.DataFrame:
        meta_df = pl.DataFrame(cols, schema=self.meta_schema, nan_to_null=True)
        self._meta_df = meta_df
        return meta_df

    def get_meta_df(self) -> pl.DataFrame:
        if self._meta_df is not None:
            return self._meta_df

        cols: Dict[str, list] = defaultdict(list)
        for frame_num in self._progress(
            range(self.first_scan_number, self.last_scan_number + 1),
            desc="read meta",
        ):
            self._read_scan_meta(frame_num, cols)

        return self._build_meta_df(cols)

    def get_frame(self, frame_num: int) -> PeakArray:
        return self._read_peaks_arrays(frame_num)

    def get_frames(self, frame_nums: Sequence[int]) -> List[PeakArray]:
        return [
            self.get_frame(int(fn))
            for fn in self._progress(frame_nums, desc="load spectra")
        ]

    def load(self) -> MassSpecData:
        scan_range = range(self.first_scan_number, self.last_scan_number + 1)
        need_meta = self._meta_df is None

        cols: Dict[str, list] = defaultdict(list) if need_meta else {}
        all_spectra: List[PeakArray] = []

        for fn in self._progress(scan_range, desc="load spectra"):
            if need_meta:
                self._read_scan_meta(fn, cols)
            all_spectra.append(self._read_peaks_arrays(fn))

        if need_meta:
            self._build_meta_df(cols)

        return MassSpecData.create(self.run_name, self._meta_df, all_spectra)
