"""
Optimized mzML reader for DelPi with targeted performance improvements.
Focus on the actual bottlenecks: XML parsing, zlib decompression, and peak processing.
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any, Sequence
import io
import re
import binascii
import zlib
from functools import lru_cache

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.cElementTree as ET

try:
    import isal.isal_zlib as fast_zlib
except ImportError:
    fast_zlib = zlib

try:
    from isal.igzip import IGzipFile as _GzipFile
except ImportError:
    from gzip import GzipFile as _GzipFile

try:
    import pybase64 as _base64_lib

    def _b64decode(binary_text: str) -> bytes:
        return _base64_lib.b64decode(binary_text, validate=False)

except ImportError:

    def _b64decode(binary_text: str) -> bytes:
        return binascii.a2b_base64(binary_text)


import numpy as np
import polars as pl

from pymsio.readers.base import MassSpecFileReader
from pymsio.readers.ms_data import MassSpecData, PeakArray, META_SCHEMA

_EMPTY_F32 = np.array([], dtype=np.float32)

# PSI-MS CV accession codes used for single-pass, order-independent parsing.
MS_LEVEL = "MS:1000511"
SCAN_TIME = "MS:1000016"
SCAN_WINDOW_LOWER = "MS:1000501"
SCAN_WINDOW_UPPER = "MS:1000500"

ISOLATION_TARGET = "MS:1000827"
ISOLATION_LOWER = "MS:1000828"
ISOLATION_UPPER = "MS:1000829"

MZ_ARRAY = "MS:1000514"
INTENSITY_ARRAY = "MS:1000515"

FLOAT32 = "MS:1000521"
FLOAT64 = "MS:1000523"
ZLIB_COMPRESSION = "MS:1000574"
NO_COMPRESSION = "MS:1000576"

MINUTE_UNIT = "UO:0000031"


def fast_process_peaks(mz_arr, int_arr):
    """Filter peaks with positive intensity. Returns (mz, ab) 1-D float32 arrays.

    Vectorized with NumPy instead of an element-by-element JIT loop: this
    avoids JIT dispatch overhead for the (typically small) per-spectrum
    arrays and includes a fast path when every intensity is already
    positive (the common case), which skips filtering entirely.
    """

    if mz_arr is None or int_arr is None or mz_arr.size == 0 or int_arr.size == 0:
        return _EMPTY_F32, _EMPTY_F32

    n = min(mz_arr.size, int_arr.size)
    if n != mz_arr.size:
        mz_arr = mz_arr[:n]
    if n != int_arr.size:
        int_arr = int_arr[:n]

    mask = int_arr > 0
    if mask.all():
        return (
            mz_arr.astype(np.float32, copy=False),
            int_arr.astype(np.float32, copy=False),
        )
    if not mask.any():
        return _EMPTY_F32, _EMPTY_F32

    return (
        mz_arr[mask].astype(np.float32, copy=False),
        int_arr[mask].astype(np.float32, copy=False),
    )


def binary_decode(
    binary_text: str, precision: int, compression: Optional[str] = None
) -> np.ndarray:
    """Optimized binary decoding with zero-copy and memory efficiency."""
    # Early validation to avoid exceptions in normal path. Avoid `.strip()`
    # here: it would scan the (potentially large) base64 payload just to
    # check for blankness; a2b_base64/pybase64 already tolerate embedded
    # whitespace/newlines.
    dtype = np.float64 if precision == 64 else np.float32
    if not binary_text:
        return np.array([], dtype=dtype)

    try:
        binary_data = _b64decode(binary_text)

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
    Optimized mzML file reader.

    Performance improvements over a naive implementation:
    - Single-pass, accession-based cvParam/binary-array extraction (no
      repeated subtree traversal).
    - Metadata-only parsing never decodes binary (m/z/intensity) arrays.
    - ``load()`` decodes peaks and collects metadata in one XML pass.
    - Peaks are written directly into preallocated flat float32 buffers
      instead of building/concatenating many small per-spectrum arrays.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        num_workers: int = 1,
        build_index: bool = False,
        index_regex: Optional[str] = None,
        buffer_size_mb: int = 8,  # Configurable buffer size in MB
    ):
        """Initialize optimized mzML reader.

        Note:
            ``num_workers`` is accepted for API compatibility but is
            currently unused: parsing is single-threaded (streaming XML
            parsing does not parallelize well due to sequential decompression
            and element ordering). It is kept as a constructor argument so
            existing callers/config do not break; it may be wired up to a
            real parallel strategy in the future if profiling justifies it.
        """
        super().__init__(file_path, num_workers)
        self._meta_df: Optional[pl.DataFrame] = None
        self._is_gzipped = self.file_path.suffix.lower() == ".gz"
        self.build_index = build_index  # For compatibility
        self.index_regex = index_regex  # For compatibility
        self.buffer_size = buffer_size_mb * 1024 * 1024
        self._num_spectra: Optional[int] = None
        self._num_spectra_resolved = False

    # ------------------------------------------------------------------
    # Fast spectrum-count extraction
    # ------------------------------------------------------------------

    _RE_OFFSET = re.compile(rb"<offset\b")
    _RE_INDEX_SPECTRUM = re.compile(
        rb'<index\s+name\s*=\s*["\']spectrum["\']', re.IGNORECASE
    )
    _RE_INDEX_END = re.compile(rb"</index\s*>")
    _RE_SPECTRUM_LIST_COUNT = re.compile(
        rb'<spectrumList\s[^>]*count\s*=\s*["\']([0-9]+)["\']'
    )

    def _count_from_index_tail(self) -> Optional[int]:
        """Read the trailing index of an indexed mzML and count <offset> entries."""
        if self._is_gzipped:
            return None
        try:
            size = self.file_path.stat().st_size
            tail_size = min(size, 1 << 20)  # last 1 MB
            with open(self.file_path, "rb") as f:
                f.seek(size - tail_size)
                tail = f.read()
            m = self._RE_INDEX_SPECTRUM.search(tail)
            if m is None:
                return None
            end = self._RE_INDEX_END.search(tail, m.end())
            block = tail[m.end() : end.start()] if end else tail[m.end() :]
            return len(self._RE_OFFSET.findall(block))
        except Exception:
            return None

    def _count_from_spectrum_list(self) -> Optional[int]:
        """Read enough of the file header to find <spectrumList count="...">."""
        try:
            read_size = 1 << 16  # 64 KB — header is small
            opener = self._open_file_handle
            with opener() as f:
                head = f.read(read_size)
            m = self._RE_SPECTRUM_LIST_COUNT.search(head)
            if m is None:
                return None
            return int(m.group(1))
        except Exception:
            return None

    @property
    def num_spectra(self) -> Optional[int]:
        if not self._num_spectra_resolved:
            self._num_spectra = (
                self._count_from_index_tail() or self._count_from_spectrum_list()
            )
            self._num_spectra_resolved = True
        return self._num_spectra

    def _open_file_handle(self):
        """Open file handle with configurable buffering for both regular and gzipped files."""
        if self._is_gzipped:
            # Wrap with a configurable large buffer (default 8MB). Uses
            # isal's IGzipFile (much faster DEFLATE) when python-isal is
            # installed, falling back to the stdlib gzip.GzipFile.
            return io.BufferedReader(
                _GzipFile(self.file_path, "rb"), buffer_size=self.buffer_size
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

    def _parse_spectrum_element(
        self, spectrum_elem, *, decode_peaks: bool
    ) -> Optional[Dict[str, Any]]:
        """Parse a single <spectrum> element in a single traversal.

        cvParam values are matched by CV accession (not by ``name``, and not
        by tracking which structural element we are nested inside) since the
        accessions used here are each unique to their context in the mzML
        schema. This lets metadata and (optionally) the mz/intensity binary
        arrays be extracted in one pass over ``spectrum_elem.iter()``,
        instead of the 3 separate subtree traversals the naive
        implementation would need (one for metadata, one to collect
        <binaryDataArray> elements, one per array for its cvParams/<binary>).

        Binary arrays are identified by their array-type cvParam accession
        (``MZ_ARRAY``/``INTENSITY_ARRAY``), not by assuming the first two
        <binaryDataArray> elements are mz/intensity - so additional arrays
        (e.g. ion mobility) before or between them are handled safely.

        When ``decode_peaks`` is False, no base64/zlib decoding happens at
        all - only cheap XML attribute access.
        """
        if spectrum_elem is None:
            return None

        spectrum_id = spectrum_elem.get("id", "")
        index_str = spectrum_elem.get("index", "0")
        index = int(index_str) if index_str.isdigit() else 0

        scan_time = 0.0
        ms_level = 1
        mz_lo = 0.0
        mz_hi = 0.0
        iso_target = None
        iso_lower = None
        iso_upper = None

        mz_binary_elem = None
        mz_precision = 32
        mz_compression = None
        intensity_binary_elem = None
        intensity_precision = 32
        intensity_compression = None

        # State for whichever <binaryDataArray> we're currently inside.
        cur_array_type: Optional[str] = None
        cur_precision = 32
        cur_compression: Optional[str] = None
        arrays_found = 0

        for elem in spectrum_elem.iter():
            tag = self._get_local_tag(elem.tag)

            if tag == "cvParam":
                accession = elem.get("accession")
                if not accession:
                    continue
                value = elem.get("value")

                if accession == MS_LEVEL:
                    if value is not None:
                        try:
                            ms_level = int(value)
                        except ValueError:
                            pass
                elif accession == SCAN_TIME:
                    if value is not None:
                        try:
                            scan_time = float(value)
                        except ValueError:
                            pass
                        else:
                            unit_acc = elem.get("unitAccession")
                            if unit_acc == MINUTE_UNIT or (
                                unit_acc is None
                                and elem.get("unitName", "minute") == "minute"
                            ):
                                scan_time *= 60.0
                elif accession == SCAN_WINDOW_LOWER:
                    if value is not None:
                        try:
                            mz_lo = float(value)
                        except ValueError:
                            pass
                elif accession == SCAN_WINDOW_UPPER:
                    if value is not None:
                        try:
                            mz_hi = float(value)
                        except ValueError:
                            pass
                elif accession == ISOLATION_TARGET:
                    if value is not None:
                        try:
                            iso_target = float(value)
                        except ValueError:
                            pass
                elif accession == ISOLATION_LOWER:
                    if value is not None:
                        try:
                            iso_lower = float(value)
                        except ValueError:
                            pass
                elif accession == ISOLATION_UPPER:
                    if value is not None:
                        try:
                            iso_upper = float(value)
                        except ValueError:
                            pass
                elif decode_peaks and arrays_found < 2:
                    if accession == MZ_ARRAY:
                        cur_array_type = "mz"
                    elif accession == INTENSITY_ARRAY:
                        cur_array_type = "intensity"
                    elif accession == FLOAT32:
                        cur_precision = 32
                    elif accession == FLOAT64:
                        cur_precision = 64
                    elif accession == ZLIB_COMPRESSION:
                        cur_compression = "zlib"
                    elif accession == NO_COMPRESSION:
                        cur_compression = None

            elif tag == "binaryDataArray":
                if decode_peaks and arrays_found < 2:
                    # Reset per-array state for this new <binaryDataArray>.
                    cur_array_type = None
                    cur_precision = 32
                    cur_compression = None

            elif tag == "binary":
                if decode_peaks and arrays_found < 2 and cur_array_type is not None:
                    if cur_array_type == "mz" and mz_binary_elem is None:
                        mz_binary_elem = elem
                        mz_precision = cur_precision
                        mz_compression = cur_compression
                        arrays_found += 1
                    elif (
                        cur_array_type == "intensity" and intensity_binary_elem is None
                    ):
                        intensity_binary_elem = elem
                        intensity_precision = cur_precision
                        intensity_compression = cur_compression
                        arrays_found += 1
                    cur_array_type = None
                    # binaryDataArrayList is the last child of <spectrum> in
                    # valid mzML, so once both required arrays are found
                    # there is nothing else left to extract.
                    if arrays_found >= 2:
                        break

        isolation_min_mz = None
        isolation_max_mz = None
        if iso_target is not None and iso_lower is not None and iso_upper is not None:
            isolation_min_mz = iso_target - iso_lower
            isolation_max_mz = iso_target + iso_upper

        mz_array = None
        intensity_array = None
        if decode_peaks:
            if mz_binary_elem is not None and mz_binary_elem.text:
                mz_array = binary_decode(
                    mz_binary_elem.text, mz_precision, mz_compression
                )
            if intensity_binary_elem is not None and intensity_binary_elem.text:
                intensity_array = binary_decode(
                    intensity_binary_elem.text,
                    intensity_precision,
                    intensity_compression,
                )

        return {
            "id": spectrum_id,
            "index": index,
            "ms_level": ms_level,
            "scan_time": scan_time,
            "mz_lo": mz_lo,
            "mz_hi": mz_hi,
            "isolation_min_mz": isolation_min_mz,
            "isolation_max_mz": isolation_max_mz,
            "mz_array": mz_array,
            "intensity_array": intensity_array,
        }

    def _spec_to_meta(self, spectrum_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert spectrum data to metadata dictionary."""
        return {
            "frame_num": np.uint32(spectrum_data["index"]),
            "mz_lo": np.float32(spectrum_data["mz_lo"]),
            "mz_hi": np.float32(spectrum_data["mz_hi"]),
            "time_in_seconds": np.float32(spectrum_data["scan_time"]),
            "ms_level": np.uint8(spectrum_data["ms_level"]),
            "isolation_min_mz": spectrum_data["isolation_min_mz"],
            "isolation_max_mz": spectrum_data["isolation_max_mz"],
        }

    def _parse_spectra(
        self,
        *,
        collect_meta: bool,
        decode_peaks: bool,
        progress=None,
    ) -> Tuple[
        List[Dict[str, Any]], List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]
    ]:
        """Single streaming pass over all <spectrum> elements.

        Args:
            collect_meta: collect one metadata row per spectrum.
            decode_peaks: decode and return the *raw* (not yet filtered or
                cast to float32) mz/intensity arrays for each spectrum. When
                ``False``, no base64/zlib decoding happens at all.
        """
        meta_rows: List[Dict[str, Any]] = []
        raw_peaks: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []

        with self._open_file_handle() as f:
            try:
                context = ET.iterparse(f, events=("end",), tag="{*}spectrum")
                use_filter = True
            except (TypeError, ValueError):
                context = ET.iterparse(f, events=("end",))
                use_filter = False

            idx = 0
            for event, elem in self._progress(
                context, progress=progress, desc="parse spectra"
            ):
                if not use_filter and self._get_local_tag(elem.tag) != "spectrum":
                    elem.clear()
                    continue

                spectrum_data = self._parse_spectrum_element(
                    elem, decode_peaks=decode_peaks
                )

                if spectrum_data is not None:
                    if collect_meta:
                        meta_rows.append(self._spec_to_meta(spectrum_data))
                    if decode_peaks:
                        raw_peaks.append(
                            (
                                spectrum_data["mz_array"],
                                spectrum_data["intensity_array"],
                            )
                        )
                else:
                    if collect_meta:
                        meta_rows.append(self._create_empty_meta(idx))
                    if decode_peaks:
                        raw_peaks.append((None, None))

                idx += 1

                # Efficient memory cleanup
                elem.clear()
                parent = elem.getparent() if hasattr(elem, "getparent") else None
                if parent is not None:
                    try:
                        parent.remove(elem)
                    except (ValueError, TypeError):
                        pass

        return meta_rows, raw_peaks

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
        """Read metadata for all spectra without decoding any binary arrays."""
        meta_rows, _ = self._parse_spectra(collect_meta=True, decode_peaks=False)
        return pl.DataFrame(meta_rows, schema=META_SCHEMA)

    def get_meta_df(self) -> pl.DataFrame:
        """Get metadata DataFrame, cached after first call."""
        if self._meta_df is None:
            self._meta_df = self._read_meta()
        return self._meta_df

    @staticmethod
    def _build_peak_array(
        raw_peaks: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
    ) -> Tuple[np.ndarray, np.ndarray, PeakArray]:
        """Filter+cast raw per-spectrum (mz, intensity) arrays directly into
        one preallocated flat float32 ``PeakArray``, instead of building one
        ``PeakArray`` per spectrum and concatenating them afterwards.

        Returns ``(peak_start, peak_stop, peaks)``.
        """
        n_spectra = len(raw_peaks)
        counts = np.zeros(n_spectra, dtype=np.uint32)
        masks: List[Optional[np.ndarray]] = [None] * n_spectra

        for i, (_, int_arr) in enumerate(raw_peaks):
            if int_arr is None or int_arr.size == 0:
                continue
            mask = int_arr > 0
            if mask.all():
                counts[i] = int_arr.size
                # No filtering needed later; leave masks[i] as None as a
                # sentinel meaning "take everything".
            else:
                masks[i] = mask
                counts[i] = int(mask.sum())

        peak_stop = np.cumsum(counts, dtype=np.uint32)
        peak_start = peak_stop - counts
        total = int(peak_stop[-1]) if n_spectra else 0
        mz_out = np.empty(total, dtype=np.float32)
        ab_out = np.empty(total, dtype=np.float32)

        for i, (mz_arr, int_arr) in enumerate(raw_peaks):
            n = int(counts[i])
            if n == 0:
                continue
            st, ed = int(peak_start[i]), int(peak_stop[i])
            mask = masks[i]
            if mask is None:
                # fast path: every peak already has positive intensity
                mz_out[st:ed] = mz_arr
                ab_out[st:ed] = int_arr
            else:
                mz_out[st:ed] = mz_arr[mask]
                ab_out[st:ed] = int_arr[mask]

        return peak_start, peak_stop, PeakArray(mz_out, ab_out)

    def load(self, progress=None) -> MassSpecData:
        """Load complete mass spectrometry data.

        Metadata and peaks are collected in the same XML pass when metadata
        has not already been loaded (i.e. this does not re-parse the file
        just because ``get_meta_df()`` wasn't called first).
        """
        need_meta = self._meta_df is None
        meta_rows, raw_peaks = self._parse_spectra(
            collect_meta=need_meta, decode_peaks=True, progress=progress
        )

        if need_meta:
            self._meta_df = pl.DataFrame(meta_rows, schema=META_SCHEMA)

        peak_start, peak_stop, peaks = self._build_peak_array(raw_peaks)

        return MassSpecData.create_from_flat(
            run_name=self.run_name,
            meta_df=self._meta_df,
            peaks=peaks,
            peak_start=peak_start,
            peak_stop=peak_stop,
        )

    def get_frame(self, frame_num: int) -> PeakArray:
        """Get peaks for a specific frame number."""
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
                spectrum_data = self._parse_spectrum_element(elem, decode_peaks=True)
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

                spectrum_data = self._parse_spectrum_element(
                    spec_elem, decode_peaks=True
                )
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
