from abc import ABC, abstractmethod
from typing import Union, List, Sequence
from pathlib import Path

from tqdm import tqdm
import polars as pl
import numpy as np

from pymsio.readers.ms_data import (
    MassSpecData,
    PeakArray,
    META_SCHEMA,
)  # noqa: F401 (re-export)

COMPRESSION_EXTENSIONS = [".gz", ".zip", ".bz2", ".xz", ".7z", ".tar"]
MS_EXTENSIONS = [".mzml", ".raw", ".d", ".wiff", ".mgf", ".mzdata", ".mz5"]


class _ProgressIterWrapper:
    """Wraps an iterable to call ``progress.update(1)`` on each step."""

    def __init__(self, iterable, progress):
        self._it = iter(iterable)
        self._bar = progress

    def __iter__(self):
        return self

    def __next__(self):
        val = next(self._it)  # propagates StopIteration naturally
        self._bar.update(1)
        return val

    def __len__(self):
        return getattr(self._it, "__length_hint__", lambda: 0)()


class MassSpecFileReader(ABC):

    meta_schema = META_SCHEMA

    def __init__(
        self,
        file_path: Union[str, Path],
        num_workers: int = 0,
    ):
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        self.file_path = file_path
        self.num_workers = num_workers

    def _progress(self, iterable, progress=None, **kwargs):
        if progress is True:
            return tqdm(iterable, **kwargs)
        elif progress:
            return _ProgressIterWrapper(iterable, progress)
        return iterable

    @staticmethod
    def extract_run_name(filepath: Union[str, Path]):

        filepath = Path(filepath)
        filename = filepath.name

        # remove compression ext
        for comp_ext in COMPRESSION_EXTENSIONS:
            if filename.lower().endswith(comp_ext):
                filename = filename[: -len(comp_ext)]
                break

        # remove ms file ext
        for ms_ext in MS_EXTENSIONS:
            if filename.lower().endswith(ms_ext):
                return filename[: -len(ms_ext)]

        return filepath.stem

    @property
    def run_name(self) -> str:
        return self.extract_run_name(self.file_path)

    @property
    def num_spectra(self):
        """Expected number of spectra, or ``None`` if unknown before parsing."""
        return None

    @abstractmethod
    def get_meta_df(self) -> pl.DataFrame:
        """
        Returns:
            pl.DataFrame: shape of (num_frames, 8)
        ┌───────────┬───────┬─────────────┬─────────────────┬──────────┬──────────────────┬──────────────────┬───────────────────┐
        │ frame_num ┆ mz_lo ┆ mz_hi       ┆ time_in_seconds ┆ ms_level ┆ isolation_min_mz ┆ isolation_max_mz ┆ isolation_win_idx │
        │ ---       ┆ ---   ┆ ---         ┆ ---             ┆ ---      ┆ ---              ┆ ---              ┆ ---               │
        │ u32       ┆ f32   ┆ f32         ┆ f32             ┆ u8       ┆ f32              ┆ f32              ┆ u32               │
        ╞═══════════╪═══════╪═════════════╪═════════════════╪══════════╪══════════════════╪══════════════════╪═══════════════════╡
        │ 1         ┆ 380.0 ┆ 980.0       ┆ 0.0             ┆ 1        ┆ null             ┆ null             ┆ null              │
        │ …         ┆ …     ┆ …           ┆ …               ┆ …        ┆ …                ┆ …                ┆ …                 │
        │ 123456    ┆ 150.0 ┆ 1816.526367 ┆ 2148.360596     ┆ 2        ┆ 878.649292       ┆ 880.650146       ┆ 249               │
        └───────────┴───────┴─────────────┴─────────────────┴──────────┴──────────────────┴──────────────────┴───────────────────┘

        """
        raise NotImplementedError()

    @abstractmethod
    def get_frame(self, frame_num: int) -> PeakArray:
        """
        Returns:
            PeakArray: NamedTuple(mz=np.ndarray[float32], ab=np.ndarray[float32])
        """
        raise NotImplementedError()

    def get_frames(self, frame_nums: Sequence[int]) -> List[PeakArray]:
        """
        Returns:
            List[PeakArray]
        """
        return [self.get_frame(fn) for fn in frame_nums]

    @abstractmethod
    def load(self, progress=None) -> MassSpecData:
        raise NotImplementedError()
