from typing import Union, List, Sequence, NamedTuple, Optional
from pathlib import Path

import h5py
import polars as pl
import numpy as np

from pymsio.readers.utils import get_frame_num_to_index_arr, compute_z_score_cdf_numba


META_SCHEMA = {
    "frame_num": pl.UInt32,
    "mz_lo": pl.Float32,
    "mz_hi": pl.Float32,
    "time_in_seconds": pl.Float32,
    "ms_level": pl.UInt8,
    "isolation_min_mz": pl.Float32,
    "isolation_max_mz": pl.Float32,
}

_EMPTY_F32 = np.array([], dtype=np.float32)


class PeakArray(NamedTuple):
    """Pair of contiguous 1-D float32 arrays (mz, ab)."""

    mz: np.ndarray
    ab: np.ndarray

    @staticmethod
    def empty() -> "PeakArray":
        return PeakArray(_EMPTY_F32, _EMPTY_F32)

    def __len__(self) -> int:
        return self.mz.shape[0]


class MassSpecData:

    thread_safe = False

    def __init__(
        self,
        run_name: str,
        meta_df: pl.DataFrame,
        peaks: PeakArray,
    ):
        self.run_name = run_name
        self.frame_num_to_index = get_frame_num_to_index_arr(meta_df["frame_num"])
        self.meta_df = meta_df
        self.peaks = peaks
        self.z_score_arr: Optional[np.ndarray] = None

    @classmethod
    def create(
        cls,
        run_name: str,
        meta_df: pl.DataFrame,
        list_of_peaks: List[PeakArray],
    ):
        assert meta_df.shape[0] == len(list_of_peaks)

        meta_df = (
            meta_df.with_columns(
                peak_count=np.asarray([len(p) for p in list_of_peaks], dtype=np.uint32)
            )
            .with_columns(pl.col("peak_count").cum_sum().alias("peak_stop"))
            .with_columns(pl.col("peak_stop").shift(1).fill_null(0).alias("peak_start"))
            .select(pl.col(list(META_SCHEMA)), pl.col("peak_start", "peak_stop"))
        )

        peaks = PeakArray(
            np.concatenate([p.mz for p in list_of_peaks]),
            np.concatenate([p.ab for p in list_of_peaks]),
        )

        return cls(run_name, meta_df, peaks)

    def compute_z_score(self):
        if self.z_score_arr is None:
            peak_range_arr = self.meta_df.select(
                pl.col("peak_start", "peak_stop")
            ).to_numpy()
            self.z_score_arr = compute_z_score_cdf_numba(self.peaks.ab, peak_range_arr)

    def get_peak_index(self, frame_num: int):
        idx = self.frame_num_to_index[frame_num]
        st = self.meta_df.item(idx, "peak_start")
        ed = self.meta_df.item(idx, "peak_stop")
        return st, ed

    def get_frame(self, frame_num: int) -> PeakArray:
        st, ed = self.get_peak_index(frame_num)
        return PeakArray(self.peaks.mz[st:ed], self.peaks.ab[st:ed])

    def get_all_peak_df(self):
        frame_num_arr = np.empty(self.peaks.mz.shape[0], dtype=np.uint32)
        for fn, st, ed in self.meta_df.select(
            pl.col("frame_num", "peak_start", "peak_stop")
        ).iter_rows():
            frame_num_arr[st:ed] = fn

        peak_df = pl.DataFrame(
            {
                "frame_num": frame_num_arr,
                "mz": self.peaks.mz,
                "ab": self.peaks.ab,
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
        mz_out = np.empty(num_peaks, dtype=np.float32)
        ab_out = np.empty(num_peaks, dtype=np.float32)
        z_out = (
            None if self.z_score_arr is None else np.empty(num_peaks, dtype=np.float32)
        )

        st = 0
        for frame_index, (frame_num, peak_st, peak_ed) in enumerate(
            peak_idx_df.iter_rows()
        ):
            n = peak_ed - peak_st
            if n > 0:
                ed = st + n
                frame_num_arr[st:ed] = frame_num
                mz_out[st:ed] = self.peaks.mz[peak_st:peak_ed]
                ab_out[st:ed] = self.peaks.ab[peak_st:peak_ed]
                if self.z_score_arr is not None:
                    z_out[st:ed] = self.z_score_arr[peak_st:peak_ed]
                st = ed

        return frame_num_arr, mz_out, ab_out, z_out

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
            hf_grp.create_dataset("mz", data=self.peaks.mz, dtype=np.float32)
            hf_grp.create_dataset("ab", data=self.peaks.ab, dtype=np.float32)

        self.meta_df.to_pandas().to_hdf(
            file_path, key=f"{group_key}/meta_df", index=False
        )
