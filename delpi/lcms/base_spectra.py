from typing import Optional, List

import polars as pl
import numpy as np

from delpi.lcms.data_container import PeakContainer, MetaContainer


def get_frame_num_to_index_arr(frame_nums: List[int]):
    num_to_idx = np.zeros(frame_nums[-1] + 1, dtype=np.uint32)
    num_to_idx[frame_nums] = np.arange(len(frame_nums), dtype=np.uint32)
    return num_to_idx


def _readonly(arr: np.ndarray) -> np.ndarray:
    """Mark an array read-only in place (no copy) for a stable numba type."""
    arr.flags.writeable = False
    return arr


class BaseSpectra:
    """Peaks and per-spectrum metadata of spectra at one MS level."""

    def __init__(self, meta_df: pl.DataFrame, peak_df: pl.DataFrame):
        assert peak_df["mz"].is_sorted(), "Peak DataFrame must be sorted by m/z"
        assert (
            meta_df["ms_level"].n_unique() == 1
        ), "All spectra must be at the same MS level"

        self.peak_df = peak_df
        self.meta_df = meta_df
        self.frame_num_to_index = get_frame_num_to_index_arr(meta_df["frame_num"])

    @property
    def ms_level(self) -> int:
        """MS level of this spectra group."""
        return self.meta_df["ms_level"].first()

    @property
    def num_frames(self) -> int:
        """Number of frames (spectra)."""
        return self.meta_df.shape[0]

    def get_peak_container(self) -> PeakContainer:
        # to_numpy() returns a readonly view or a writable copy depending on
        # chunking; all read-only gives numba one array type (njit never writes)
        return PeakContainer(
            frame_num_arr=_readonly(self.peak_df["frame_num"].to_numpy()),
            mz_arr=_readonly(self.peak_df["mz"].to_numpy()),
            ab_arr=_readonly(self.peak_df["ab"].to_numpy()),
            z_score_arr=_readonly(self.peak_df["z_score"].to_numpy()),
        )

    def get_meta_container(self) -> MetaContainer:
        meta_df = self.meta_df
        return MetaContainer(
            frame_num_arr=_readonly(meta_df["frame_num"].to_numpy()),
            frame_num_to_index_arr=self.frame_num_to_index,
            ms_level=meta_df.item(0, "ms_level"),
            rt_arr=_readonly(meta_df["time_in_seconds"].to_numpy()),
            isolation_min_mz_arr=_readonly(meta_df["isolation_min_mz"].to_numpy()),
            isolation_max_mz_arr=_readonly(meta_df["isolation_max_mz"].to_numpy()),
        )

    def get_xic(self, mz: float, tolerance_in_ppm: float = 10) -> pl.DataFrame:
        """XIC: all peaks within tolerance_in_ppm of mz."""
        experimental_peak_df = self.peak_df
        mz_tol = mz * tolerance_in_ppm * 1e-6

        exp_start, exp_stop = experimental_peak_df.select(
            pl.col("mz")
            .search_sorted(mz - mz_tol, "left")
            .alias("exp_start"),  # inclusive
            pl.col("mz")
            .search_sorted(mz + mz_tol, "right")
            .alias("exp_stop"),  # exclusive
        ).row(0)

        return experimental_peak_df[exp_start:exp_stop]

    def get_spectrum(self, frame_num: int) -> pl.DataFrame:
        """All peaks of the spectrum with frame_num."""
        return self.peak_df.filter(pl.col("frame_num") == frame_num)

    def match(self, theoretical_peak_df, tolerance_in_ppm):
        """Per theoretical mz, the [exp_start, exp_stop) peak_df rows in tolerance."""
        experimental_peak_df = self.peak_df
        mz_arr = theoretical_peak_df["mz"]
        mz_tol = mz_arr * tolerance_in_ppm * 1e-6

        theo_to_exp_df = experimental_peak_df.select(
            pl.col("mz")
            .search_sorted(mz_arr - mz_tol, "left")
            .alias("exp_start"),  # inclusive
            pl.col("mz")
            .search_sorted(mz_arr + mz_tol, "right")
            .alias("exp_stop"),  # exclusive
        )

        return theo_to_exp_df

    @staticmethod
    def make_peak_df(
        frame_num_arr: np.ndarray,
        mz_arr: np.ndarray,
        ab_arr: np.ndarray,
        z_score_arr: Optional[np.ndarray] = None,
    ):
        """Peak DataFrame sorted by mz; z_score is added only if given."""
        data = {
            "frame_num": frame_num_arr,
            "mz": mz_arr,
            "ab": ab_arr,
        }

        if z_score_arr is not None:
            data["z_score"] = z_score_arr

        return pl.DataFrame(data).sort(pl.col("mz"))
