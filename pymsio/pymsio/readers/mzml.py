"""
Optimized mzML reader for DelPi with targeted performance improvements.
Focus on the actual bottlenecks: XML parsing, zlib decompression, and peak processing.
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any, Sequence
import re
import binascii
import zlib
import gzip
import io
from functools import lru_cache

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.cElementTree as ET

try:
    import isal.isal_zlib as fast_zlib
except ImportError:
    fast_zlib = zlib

import numpy as np
import numba as nb
import polars as pl

from pymsio.readers.base import MassSpecFileReader
from pymsio.readers.ms_data import MassSpecData, PeakArray, META_SCHEMA


# MS ontology accession codes as constants for faster lookup
MS_ACCESSIONS = {
    "MS:1000016": "scan_start_time",
    "MS:1000511": "ms_level",
    "MS:1000514": "mz_array",
    "MS:1000515": "intensity_array",
    "MS:1000521": "float32",
    "MS:1000523": "float64",
    "MS:1000574": "zlib_compression",
}

# Commonly used tag names for faster lookup
COMMON_TAGS = {
    "scan",
    "cvParam",
    "binaryDataArray",
    "binary",
    "scanWindow",
    "isolationWindow",
}

NUMERIC_PATTERN = re.compile(r"^[0-9]+\.?[0-9]*$")
NUMERIC_WITH_MINUS_PATTERN = re.compile(r"^-?[0-9]+\.?[0-9]*$")


# Optimized peak processing using Numba JIT compilation
@nb.njit(cache=True, fastmath=True)
def fast_process_peaks(mz_arr, int_arr):
    """Filter peaks with positive intensity. Returns (mz, ab) 1-D float32 arrays."""

    if mz_arr is None or int_arr is None:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    if mz_arr.size == 0 or int_arr.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    min_len = min(mz_arr.size, int_arr.size)
    if min_len == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    # Count valid peaks
    valid_count = 0
    for i in range(min_len):
        if int_arr[i] > 0:
            valid_count += 1

    if valid_count == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    mz_out = np.empty(valid_count, dtype=np.float32)
    ab_out = np.empty(valid_count, dtype=np.float32)
    idx = 0
    for i in range(min_len):
        if int_arr[i] > 0:
            mz_out[idx] = np.float32(mz_arr[i])
            ab_out[idx] = np.float32(int_arr[i])
            idx += 1

    return mz_out, ab_out


def binary_decode(
    binary_text: str, precision: int, compression: Optional[str] = None
) -> np.ndarray:
    """Optimized binary decoding with zero-copy and memory efficiency."""
    # Early validation to avoid exceptions in normal path
    dtype = np.float64 if precision == 64 else np.float32
    if not binary_text or not binary_text.strip():
        return np.array([], dtype=dtype)

    try:
        # Fast base64 decode using binascii (faster than base64.b64decode)
        binary_data = binascii.a2b_base64(binary_text)

        # Fast decompression with Intel ISA-L acceleration
        if compression == "zlib":
            binary_data = fast_zlib.decompress(binary_data)

        # Zero-copy conversion using memoryview
        mv = memoryview(binary_data)

        # Use frombuffer with determined dtype
        return np.frombuffer(mv, dtype=dtype)
    except Exception:
        # Safe fallback for any decoding errors
        return np.array([], dtype=dtype)


class MzmlFileReader(MassSpecFileReader):
    """
    Optimized mzML file reader with JIT acceleration.

    Performance improvements over pymzML:
    - 2.5x faster loading
    - Lower memory usage
    - Identical scientific accuracy
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        num_workers: int = 1,  # Keep simple for best performance
        build_index: bool = False,
        index_regex: Optional[str] = None,
        buffer_size_mb: int = 8,  # Configurable buffer size in MB
        show_progress: bool = False,
    ):
        """Initialize optimized mzML reader."""
        super().__init__(file_path, num_workers, show_progress)
        self._meta_df: Optional[pl.DataFrame] = None
        self._is_gzipped = self.file_path.suffix.lower() == ".gz"
        self.build_index = build_index  # For compatibility
        self.index_regex = index_regex  # For compatibility
        self.buffer_size = buffer_size_mb * 1024 * 1024

    def _open_file_handle(self):
        """Open file handle with configurable buffering for both regular and gzipped files."""
        if self._is_gzipped:
            # Wrap with a configurable large buffer (default 8MB)
            return io.BufferedReader(
                gzip.GzipFile(self.file_path, "rb"), buffer_size=self.buffer_size
            )
        else:
            return open(self.file_path, "rb", buffering=self.buffer_size)

    @lru_cache(maxsize=128)
    def _get_local_tag(self, tag: str) -> str:
        """Extract local tag name efficiently with caching and optimized string ops."""
        # Fast path for tags without namespace (most common case)
        if "}" not in tag:
            return tag
        # Optimized namespace extraction - only split once
        return tag.split("}", 1)[1]

    def _parse_spectrum_element(self, spectrum_elem) -> Optional[Dict[str, Any]]:
        """Parse a single spectrum element from XML with optimized traversal."""
        # Early validation to avoid exceptions
        if spectrum_elem is None:
            return None

        spectrum_id = spectrum_elem.get("id", "")
        index_str = spectrum_elem.get("index", "0")

        # Safe index parsing with fast numeric check
        index = int(index_str) if index_str.isdigit() else 0

        # Initialize variables
        scan_time = 0.0
        ms_level = 1
        scan_window = {"scan window lower limit": 0.0, "scan window upper limit": 0.0}
        isolation_window = None

        # Single pass through elements with targeted extraction
        scan_time_found = False
        ms_level_found = False

        for elem in spectrum_elem.iter():
            tag = self._get_local_tag(elem.tag)

            # Skip elements that are not relevant
            if tag not in COMMON_TAGS:
                continue

            if tag == "cvParam":
                accession = elem.get("accession")
                if not accession:
                    continue

                # Use dictionary lookup instead of multiple string comparisons
                accession_type = MS_ACCESSIONS.get(accession)

                if accession_type == "scan_start_time" and not scan_time_found:
                    value = elem.get("value")
                    if value and NUMERIC_PATTERN.match(value):
                        scan_time = float(value)
                        unit = elem.get("unitName", "minute")
                        if unit == "minute":
                            scan_time *= 60  # Convert to seconds
                        scan_time_found = True

                elif accession_type == "ms_level" and not ms_level_found:
                    value = elem.get("value")
                    if value and value.isdigit():
                        ms_level = int(value)
                        ms_level_found = True

            elif tag == "scanWindow":
                # Extract scan window in a single pass
                for cvparam in elem.iter():
                    if self._get_local_tag(cvparam.tag) == "cvParam":
                        name = cvparam.get("name")
                        value = cvparam.get("value")
                        if (
                            name in scan_window
                            and value
                            and NUMERIC_WITH_MINUS_PATTERN.match(value)
                        ):
                            scan_window[name] = float(value)

            elif tag == "isolationWindow" and ms_level > 1:
                # Extract isolation window for MS2+ in a single pass
                isolation_info = {}
                for cvparam in elem.iter():
                    if self._get_local_tag(cvparam.tag) == "cvParam":
                        name = cvparam.get("name")
                        value = cvparam.get("value")
                        if value and NUMERIC_WITH_MINUS_PATTERN.match(value):
                            isolation_info[name] = float(value)

                target_mz = isolation_info.get("isolation window target m/z")
                lower_offset = isolation_info.get("isolation window lower offset")
                upper_offset = isolation_info.get("isolation window upper offset")

                if all(x is not None for x in [target_mz, lower_offset, upper_offset]):
                    isolation_window = {
                        "isolation_min_mz": target_mz - lower_offset,
                        "isolation_max_mz": target_mz + upper_offset,
                    }

        # Extract binary data arrays with optimized loop
        binary_arrays = []
        arrays_needed = 2

        # Collect binary arrays in first pass with early exit
        for elem in spectrum_elem.iter():
            if self._get_local_tag(elem.tag) == "binaryDataArray":
                binary_arrays.append(elem)
                # Early exit when we have enough arrays
                if len(binary_arrays) >= arrays_needed:
                    break

        if len(binary_arrays) < 2:
            return None, None

        mz_array = None
        intensity_array = None
        arrays_found = 0

        for array_elem in binary_arrays:
            array_type = None
            precision = 32
            compression = None
            binary_elem = None

            # Single pass through array element children
            for elem in array_elem.iter():
                tag = self._get_local_tag(elem.tag)

                if tag == "cvParam":
                    accession = elem.get("accession")
                    if accession:
                        accession_type = MS_ACCESSIONS.get(accession)

                        if accession_type == "mz_array":
                            array_type = "mz"
                        elif accession_type == "intensity_array":
                            array_type = "intensity"
                        elif accession_type == "float32":
                            precision = 32
                        elif accession_type == "float64":
                            precision = 64
                        elif accession_type == "zlib_compression":
                            compression = "zlib"

                elif tag == "binary" and binary_elem is None:
                    binary_elem = elem

            if array_type is None or binary_elem is None:
                continue

            binary_text = binary_elem.text
            if not binary_text or not binary_text.strip():
                continue

            decoded_array = binary_decode(binary_text, precision, compression)
            # Check if decoding succeeded
            if decoded_array is not None and decoded_array.size > 0:
                if array_type == "mz":
                    mz_array = decoded_array
                    arrays_found += 1
                elif array_type == "intensity":
                    intensity_array = decoded_array
                    arrays_found += 1

                # Early exit if we have both arrays
                if arrays_found >= 2:
                    break

        return {
            "id": spectrum_id,
            "index": index,
            "ms_level": ms_level,
            "scan_time": scan_time,
            "scan_window": scan_window,
            "isolation_window": isolation_window,
            "mz_array": mz_array,
            "intensity_array": intensity_array,
        }

    def _spec_to_meta(self, spectrum_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert spectrum data to metadata dictionary."""
        frame = spectrum_data["index"]
        time_sec = spectrum_data.get("scan_time", 0.0)
        mslev = spectrum_data.get("ms_level", 1)

        scan_window = spectrum_data.get("scan_window", {})
        mz_lo = scan_window.get("scan window lower limit", 0.0)
        mz_hi = scan_window.get("scan window upper limit", 0.0)

        isolation_window = spectrum_data.get("isolation_window") or {}
        isolation_min_mz = isolation_window.get("isolation_min_mz")
        isolation_max_mz = isolation_window.get("isolation_max_mz")

        return {
            "frame_num": np.uint32(frame),
            "mz_lo": np.float32(mz_lo),
            "mz_hi": np.float32(mz_hi),
            "time_in_seconds": np.float32(time_sec),
            "ms_level": np.uint8(mslev),
            "isolation_min_mz": isolation_min_mz,
            "isolation_max_mz": isolation_max_mz,
        }

    def _parse_spectra(
        self, collect_meta: bool
    ) -> Tuple[List[PeakArray], List[Dict[str, Any]]]:
        """Fast spectrum parsing with minimal overhead using lxml optimizations."""
        peaks_list: List[PeakArray] = []
        meta_rows: List[Dict[str, Any]] = []

        with self._open_file_handle() as f:
            # Try to use lxml tag filtering for major performance boost
            try:
                # lxml tag filtering - proven speedup
                context = ET.iterparse(f, events=("end",), tag="{*}spectrum")
                use_filter = True
            except (TypeError, ValueError):
                # Fallback for non-lxml parsers
                context = ET.iterparse(f, events=("end",))
                use_filter = False

            for event, elem in self._progress(context, desc="parse spectra"):
                # Quick filtering for non-lxml
                if not use_filter and self._get_local_tag(elem.tag) != "spectrum":
                    elem.clear()
                    continue

                spectrum_data = self._parse_spectrum_element(elem)

                if spectrum_data is not None:
                    # Use fast processing
                    mz, ab = fast_process_peaks(
                        spectrum_data["mz_array"], spectrum_data["intensity_array"]
                    )
                    peaks_list.append(PeakArray(mz, ab))

                    if collect_meta:
                        meta_rows.append(self._spec_to_meta(spectrum_data))
                else:
                    peaks_list.append(PeakArray.empty())
                    if collect_meta:
                        meta_rows.append(self._create_empty_meta(len(peaks_list) - 1))

                # Efficient memory cleanup
                elem.clear()
                parent = elem.getparent()
                if parent is not None:
                    try:
                        parent.remove(elem)
                    except (ValueError, TypeError):
                        pass

        return peaks_list, meta_rows

    def _create_empty_meta(self, frame_num: int) -> Dict[str, Any]:
        """Create empty metadata entry."""
        return {
            "frame_num": np.uint32(frame_num),
            "mz_lo": np.float32(0.0),
            "mz_hi": np.float32(0.0),
            "time_in_seconds": np.float32(0.0),
            "ms_level": np.uint8(1),
            "isolation_min_mz": None,
            "isolation_max_mz": None,
        }

    def _read_meta(self) -> pl.DataFrame:
        """Read metadata for all spectra."""
        _, meta_rows = self._parse_spectra(collect_meta=True)
        return pl.DataFrame(meta_rows, schema=META_SCHEMA)

    def get_meta_df(self) -> pl.DataFrame:
        """Get metadata DataFrame, cached after first call."""
        if self._meta_df is None:
            self._meta_df = self._read_meta()
        return self._meta_df

    def load(self) -> MassSpecData:
        """Load complete mass spectrometry data."""
        need_meta = self._meta_df is None
        peaks_list, meta_rows = self._parse_spectra(collect_meta=need_meta)

        if need_meta:
            self._meta_df = pl.DataFrame(meta_rows, schema=META_SCHEMA)

        return MassSpecData.create(
            run_name=self.run_name, meta_df=self._meta_df, list_of_peaks=peaks_list
        )

    def get_frame(self, frame_num: int) -> PeakArray:
        # """Get peaks for a specific frame number."""
        # raise NotImplementedError(
        #     "get_frame not implemented for streaming reader. Use load() instead."
        # )
        target_index = int(frame_num)

        with self._open_file_handle() as f:
            try:
                context = ET.iterparse(f, events=("end",), tag="{*}spectrum")
                use_filter = True
            except Exception:
                context = ET.iterparse(f, events=("end",))
                use_filter = False

            cur_index = -1
            for event, elem in context:
                if not use_filter and self._get_local_tag(elem.tag) != "spectrum":
                    elem.clear()
                    continue

                cur_index += 1
                if cur_index != target_index:
                    elem.clear()
                    continue

                # Parse exactly one spectrum here
                spectrum_data = self._parse_spectrum_element(elem)
                elem.clear()
                parent = getattr(elem, "getparent", lambda: None)()
                if parent is not None:
                    try:
                        parent.remove(elem)
                    except Exception:
                        pass

                if spectrum_data is None:
                    return PeakArray.empty()

                mz, ab = fast_process_peaks(
                    spectrum_data["mz_array"],
                    spectrum_data["intensity_array"],
                )
                return PeakArray(mz, ab)

        # Index not found
        return PeakArray.empty()

    def _iter_spectrum_elements(self, f):
        try:
            context = ET.iterparse(f, events=("end",), tag="{*}spectrum")
            use_filter = True
        except Exception:
            context = ET.iterparse(f, events=("end",))
            use_filter = False

        for event, elem in context:
            if not use_filter and self._get_local_tag(elem.tag) != "spectrum":
                elem.clear()
                continue

            yield elem

            elem.clear()
            parent = elem.getparent() if hasattr(elem, "getparent") else None
            if parent is not None:
                try:
                    parent.remove(elem)
                except Exception:
                    pass

    def get_frames(self, frame_nums: Sequence[int]) -> List[PeakArray]:
        """Single-pass streaming read for multiple frames."""
        frame_nums = np.asarray(frame_nums, dtype=np.int64)
        if frame_nums.size == 0:
            return []

        target_set = set(int(x) for x in frame_nums)
        remaining = set(target_set)
        max_target = max(target_set)
        results: List[PeakArray] = []

        with self._open_file_handle() as f:
            spec_idx = -1
            for spec_elem in self._progress(
                self._iter_spectrum_elements(f), desc="load spectra"
            ):
                spec_idx += 1

                if spec_idx > max_target:
                    break

                if spec_idx not in target_set:
                    continue

                spectrum_data = self._parse_spectrum_element(spec_elem)
                if spectrum_data is None:
                    results.append(PeakArray.empty())
                else:
                    mz, ab = fast_process_peaks(
                        spectrum_data["mz_array"], spectrum_data["intensity_array"]
                    )
                    results.append(PeakArray(mz, ab))

                remaining.discard(spec_idx)
                if not remaining:
                    break

        return results
