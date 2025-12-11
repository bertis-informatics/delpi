from abc import ABC, abstractmethod
from typing import Union, List, Sequence
from pathlib import Path
import h5py

import torch
from torch.distributions.normal import Normal
import polars as pl
import numpy as np

from delpi.lcms.base_spectra import get_frame_num_to_index_arr
from delpi.utils.peak import compute_z_score


torch_norm = Normal(loc=0, scale=1.0)

META_SCHEMA = {
    "frame_num": pl.UInt32,
    "mz_lo": pl.Float32,
    "mz_hi": pl.Float32,
    "time_in_seconds": pl.Float32,
    "ms_level": pl.UInt8,
    "isolation_min_mz": pl.Float32,
    "isolation_max_mz": pl.Float32,
}

COMPRESSION_EXTENSIONS = [".gz", ".zip", ".bz2", ".xz", ".7z", ".tar"]
MS_EXTENSIONS = [".mzml", ".raw", ".d", ".wiff", ".mgf", ".mzdata", ".mz5"]


class MassSpecData:

    thread_safe = False

    def __init__(self, run_name: str, meta_df: pl.DataFrame, peak_arr: np.ndarray):

        self.run_name = run_name
        self.frame_num_to_index = get_frame_num_to_index_arr(meta_df["frame_num"])
        self.meta_df = meta_df
        self.peak_arr = peak_arr
        self.z_score_arr = None

    @classmethod
    def create(
        cls, run_name: str, meta_df: pl.DataFrame, list_of_peaks: List[np.ndarray]
    ):

        assert meta_df.shape[0] == len(list_of_peaks)

        meta_df = (
            meta_df.with_columns(
                peak_count=np.asarray(
                    [peaks.shape[0] for peaks in list_of_peaks], dtype=np.uint32
                )
            )
            .with_columns(pl.col("peak_count").cum_sum().alias("peak_stop"))
            .with_columns(pl.col("peak_stop").shift(1).fill_null(0).alias("peak_start"))
            .select(pl.col(list(META_SCHEMA)), pl.col("peak_start", "peak_stop"))
        )
        peak_arr = np.concatenate(list_of_peaks, axis=0)

        return cls(run_name, meta_df, peak_arr)

    def compute_z_score(self):

        if self.z_score_arr is None:
            peak_arr = self.peak_arr
            peak_range_arr = self.meta_df.select(
                pl.col("peak_start", "peak_stop")
            ).to_numpy()

            # setting z_score
            z_score = compute_z_score(peak_arr, peak_range_arr)
            z_score = torch_norm.cdf(torch.from_numpy(z_score)).numpy()
            self.z_score_arr = z_score

    def get_peak_index(self, frame_num: int):
        idx = self.frame_num_to_index[frame_num]
        st = self.meta_df.item(idx, "peak_start")
        ed = self.meta_df.item(idx, "peak_stop")
        return st, ed

    def get_peaks(self, frame_num: int):

        idx = self.frame_num_to_index[frame_num]
        st = self.meta_df.item(idx, "peak_start")
        ed = self.meta_df.item(idx, "peak_stop")

        return self.peak_arr[st:ed]

    def get_all_peak_df(self):
        frame_num_arr = np.empty(self.peak_arr.shape[0], dtype=np.uint32)
        for fn, st, ed in self.meta_df.select(
            pl.col("frame_num", "peak_start", "peak_stop")
        ).iter_rows():
            frame_num_arr[st:ed] = fn

        peak_df = pl.DataFrame(
            {
                "frame_num": frame_num_arr,
                "mz": self.peak_arr[:, 0],
                "ab": self.peak_arr[:, 1],
            }
        )
        return peak_df

    def collect_peaks(self, frame_nums: Sequence[int]):

        peak_idx_df = self.meta_df[self.frame_num_to_index[frame_nums]].select(
            pl.col("frame_num", "peak_start", "peak_stop")
        )

        num_peaks = peak_idx_df.select(
            (pl.col("peak_stop") - pl.col("peak_start")).alias("num_peaks")
        )["num_peaks"].sum()

        frame_num_arr = np.empty(num_peaks, dtype=np.uint32)
        mz_arr = np.empty(num_peaks, dtype=np.float32)
        ab_arr = np.empty(num_peaks, dtype=np.float32)
        z_arr = (
            None if self.z_score_arr is None else np.empty(num_peaks, dtype=np.float32)
        )
        # frame_index_arr = np.empty(num_peaks, dtype=np.uint32)

        st = 0
        for frame_index, (frame_num, peak_st, peak_ed) in enumerate(
            peak_idx_df.iter_rows()
        ):
            peaks = self.peak_arr[peak_st:peak_ed, :]
            ed = st + peaks.shape[0]
            if peaks.shape[0] > 0:
                # frame_index_arr[st:ed] = frame_index
                frame_num_arr[st:ed] = frame_num
                mz_arr[st:ed] = peaks[:, 0]
                ab_arr[st:ed] = peaks[:, 1]
                if self.z_score_arr is not None:
                    z_arr[st:ed] = self.z_score_arr[peak_st:peak_ed]
            st = ed

        return frame_num_arr, mz_arr, ab_arr, z_arr

    def write_hdf(
        self,
        file_path: Union[str, Path],
        overwrite: bool = False,
    ):
        file_path = Path(file_path)
        group_key = self.run_name

        with h5py.File(file_path, "a") as hf:
            if group_key in hf:
                if overwrite:
                    del hf[group_key]
                else:
                    raise FileExistsError("LC/MS data already exists")
            hf_grp = hf.create_group(group_key)
            _ = hf_grp.create_dataset("peak", data=self.peak_arr, dtype=np.float32)

        self.meta_df.to_pandas().to_hdf(
            file_path, key=f"{group_key}/meta_df", index=False
        )


class MassSpecFileReader(ABC):

    meta_schema = META_SCHEMA

    def __init__(self, file_path: Union[str, Path], num_workers: int = 0):
        self.file_path = Path(file_path)
        self.num_workers = num_workers

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
    def get_frame(self, frame_num: int) -> np.ndarray:
        """
        Returns:
            np.ndarray: dtype=[('mz', np.float32), ('ab', np.float32)]
        """
        raise NotImplementedError()

    def get_frames(self, frame_nums: Sequence[int]) -> List[np.ndarray]:
        """
        Returns:
            List[np.ndarray: dtype=[('mz', np.float32), ('ab', np.float32)]]
        """
        return [self.get_frame(fn) for fn in frame_nums]

    @abstractmethod
    def load(self) -> MassSpecData:
        raise NotImplementedError()
