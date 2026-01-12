import os
from tqdm import trange
from pathlib import Path
from collections import defaultdict

import numpy as np
import numba as nb
import polars as pl
from pathlib import Path
from typing import Sequence, Union, List, Iterator

from delpi.lcms.reader.base import MassSpecFileReader, MassSpecData

dll_dir = Path(os.path.join(os.path.dirname(__file__), "dlls"))
if not dll_dir.exists():
    raise ValueError(f"Thermo DLL directory not found: {dll_dir}")

try:
    import clr

    clr.AddReference("System")
    import System

    from delpi.utils.clr import DotNetArrayToNPArray 

    # DLL 로드
    clr.AddReference(os.path.join(dll_dir, "thermo_fisher/ThermoFisher.CommonCore.Data.dll"))
    clr.AddReference(os.path.join(dll_dir, "thermo_fisher/ThermoFisher.CommonCore.RawFileReader.dll"))

    import ThermoFisher
    from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
    from ThermoFisher.CommonCore.Data.Interfaces import IScanEvent, IScanEventBase

    LOADED_DLL = True
except Exception:
    LOADED_DLL = False

print("Load_dlls", LOADED_DLL)

def _parse_mono_and_charge(trailer_data):
    labels = trailer_data.Labels
    values = trailer_data.Values

    mono_mz = None
    charge = None

    for i in range(labels.Length):
        label = labels[i]
        if label == "Monoisotopic M/Z:":
            v = values[i]
            if isinstance(v, str):
                mono_mz = float(v.strip()) if v.strip() else None
            else:
                mono_mz = float(v)
        elif label == "Charge State:":
            v = values[i]
            if isinstance(v, str):
                charge = int(v.strip()) if v.strip() else None
            else:
                charge = int(v)

        if mono_mz is not None and charge is not None:
            break

    return mono_mz, charge


# Optimized peak processing using Numba JIT compilation
@nb.njit(cache=True, fastmath=True)
def fast_process_peaks(mz_arr, int_arr):

    if mz_arr is None or int_arr is None:
        return np.empty((0, 2), dtype=np.float32)

    """Ultra-fast Numba JIT compiled version - very clean and fast code."""
    # Input validation
    if mz_arr.size == 0 or int_arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Handle size mismatch
    min_len = min(mz_arr.size, int_arr.size)
    if min_len == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Count valid peaks (JIT makes this loop very fast)
    valid_count = 0
    for i in range(min_len):
        if int_arr[i] > 0:
            valid_count += 1

    if valid_count == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Extract valid peaks (simple and fast with JIT)
    result = np.empty((valid_count, 2), dtype=np.float32)
    idx = 0
    for i in range(min_len):
        if int_arr[i] > 0:
            result[idx, 0] = np.float32(mz_arr[i])
            result[idx, 1] = np.float32(int_arr[i])
            idx += 1

    return result


class ThermoRawReader(MassSpecFileReader):
    thread_safe = False  # Just match it with BMSIO Reader (True/False adjustment if necessary)

    def __init__(self, 
                filepath: Union[str, Path],
                num_workers: int = 0,
                centroided : bool = True,
                dda: bool = False,):
        if not LOADED_DLL:
            raise ValueError("ERROR DLL import")
        
        super().__init__(filepath, num_workers)
        
        self.centroided = centroided
        self.dda = dda

        self.filepath = str(filepath)
        self._raw = RawFileReaderAdapter.FileFactory(self.filepath)
        self._raw.SelectInstrument(ThermoFisher.CommonCore.Data.Business.Device.MS, 1)

        self._meta_df: pl.DataFrame | None = None

    def close(self):
        if self._raw is not None:
            self._raw.Dispose()
            self._raw = None

    @property
    def acquisition_date(self) -> str:
        return self._raw.CreationDate.ToString("o")
    
    @property
    def num_frames(self) -> int:
        return self._raw.RunHeaderEx.LastSpectrum - self._raw.RunHeaderEx.FirstSpectrum + 1

    @property
    def first_scan_number(self) -> int:
        return self._raw.RunHeaderEx.FirstSpectrum

    @property
    def last_scan_number(self) -> int:
        return self._raw.RunHeaderEx.LastSpectrum

    @property
    def instrument(self) -> str:
        # to do
        # return self._raw.GetAllInstrumentNamesFromInstrumentMethod()
        return System.String.Join(
            " -> ", self._raw.GetAllInstrumentNamesFromInstrumentMethod()
        )
    
    def _read_peaks_arrays(self, frame_num: int, prefer_centroid: bool | None = None):
        
        if prefer_centroid is None:
            prefer_centroid = self.centroided 

        is_centroid = self._raw.IsCentroidScanFromScanNumber(frame_num)

        if (not is_centroid) and prefer_centroid:
            # Scan is profile, but user wanted centroid
            centroids = self._raw.GetSimplifiedCentroids(frame_num)  # ISimpleScanAccess
            mz_arr = DotNetArrayToNPArray(centroids.Masses)
            inten_arr = DotNetArrayToNPArray(centroids.Intensities)
        else:
            # return data as-is
            data = self._raw.GetSimplifiedScan(frame_num)  # ISimpleScanAccess
            mz_arr = DotNetArrayToNPArray(data.Masses)
            inten_arr = DotNetArrayToNPArray(data.Intensities)

        return mz_arr, inten_arr
    
    def get_meta_df(self) -> pl.DataFrame:
        if self._meta_df is not None:
            return self._meta_df
        
        min_frame = self.first_scan_number
        max_frame = self.last_scan_number

        rows: list[dict] = []

        for frame_num in trange(
            min_frame, max_frame + 1, desc="Reading Thermo meta"
        ):
            scan_stats = self._raw.GetScanStatsForScanNumber(frame_num)
            scan_event = IScanEventBase(
                self._raw.GetScanEventForScanNumber(frame_num)
            )

            try:
                rt = float(scan_stats.StartTime)  # minute
            except AttributeError:
                rt = float(self._raw.RetentionTimeFromScanNumber(frame_num))

            ms_level = int(scan_event.MSOrder)

            if ms_level == 1:
                isolation_min_mz = np.nan
                isolation_max_mz = np.nan
            else:
                isolation_center = scan_event.GetReaction(0).PrecursorMass
                isolation_width = scan_event.GetReaction(0).IsolationWidth

                if self.dda:
                    trailer_data = self._raw.GetTrailerExtraInformation(
                        frame_num
                    )
                    mono_mz, charge = _parse_mono_and_charge(trailer_data)

                    if mono_mz is None or mono_mz <= 0:
                        mono_mz = isolation_center
                    if charge is None:
                        charge = 0
                else:
                    mono_mz = isolation_center
                    charge = 0

                if mono_mz <= 0:
                    ms_level = 1
                    isolation_min_mz = np.nan
                    isolation_max_mz = np.nan
                else:
                    isolation_min_mz = isolation_center - isolation_width / 2.0
                    isolation_max_mz = isolation_center + isolation_width / 2.0
            
            
            mz_lo = float(scan_stats.LowMass)
            mz_hi = float(scan_stats.HighMass)

            rows.append(
                {
                    "frame_num": frame_num,
                    "time_in_seconds": rt * 60,
                    "ms_level": ms_level,
                    "isolation_min_mz": isolation_min_mz,
                    "isolation_max_mz": isolation_max_mz,
                    "mz_lo": mz_lo,
                    "mz_hi": mz_hi,
                }
            )

        meta_df = pl.DataFrame(rows)

        meta_df = meta_df.with_columns(
            [
                pl.col("frame_num").cast(pl.UInt32),
                pl.col("mz_lo").cast(pl.Float32),
                pl.col("mz_hi").cast(pl.Float32),
                pl.col("time_in_seconds").cast(pl.Float32),
                pl.col("ms_level").cast(pl.UInt8),
                pl.col("isolation_min_mz").cast(pl.Float32),
                pl.col("isolation_max_mz").cast(pl.Float32),
            ]
        )

        self._meta_df = meta_df
        return meta_df
    
    def get_frame(self, frame_num: int) -> np.ndarray:
        mz_arr, inten_arr = self._read_peaks_arrays(frame_num)

        return fast_process_peaks(mz_arr, inten_arr)
    
    def get_frames(self, frame_nums: Sequence[int]) -> List[np.ndarray]:
        return list(self.iter_frames(frame_nums, desc="Reading Thermo frames"))
    
    def iter_frames(self, frame_nums: Sequence[int], desc: str = "Reading Thermo frames"
                ) -> Iterator[np.ndarray]:
        frame_nums = np.asarray(frame_nums, dtype=np.int32)

        for i in trange(len(frame_nums), desc=desc):
            fn = int(frame_nums[i])
            yield self.get_frame(fn)
    
    def load(self) -> MassSpecData:
        meta_df = self.get_meta_df()

        min_frame = int(meta_df["frame_num"].min())
        max_frame = int(meta_df["frame_num"].max())

        batch_size = 1024 * 10
        batch_ranges = [
            (start, min(start + batch_size - 1, max_frame))
            for start in range(min_frame, max_frame + 1, batch_size)
        ]

        all_spectra: List[np.ndarray] = []

        for i in trange(len(batch_ranges), desc="load spectra"):
            min_fr, max_fr = batch_ranges[i]
            frame_nums = range(min_fr, max_fr + 1)

            for peaks in self.iter_frames(frame_nums):
                all_spectra.append(peaks)

        return MassSpecData.create(self.run_name, meta_df, all_spectra)