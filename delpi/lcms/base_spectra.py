from typing import Optional, List

import polars as pl
import numpy as np

from delpi.lcms.data_container import PeakContainer, MetaContainer


def get_frame_num_to_index_arr(frame_nums: List[int]):
    num_to_idx = np.zeros(frame_nums[-1] + 1, dtype=np.uint32)
    num_to_idx[frame_nums] = np.arange(len(frame_nums), dtype=np.uint32)
    return num_to_idx


class BaseSpectra:
    """
    Base class for handling mass spectrometry spectra groups.

    This class provides common functionality for managing groups of spectra
    at different MS levels, including peak data, metadata, and analysis methods.
    """

    def __init__(self, meta_df: pl.DataFrame, peak_df: pl.DataFrame):
        """
        Initialize BaseSpectra.

        Args:
            meta_df: Metadata DataFrame containing spectrum information
            peak_df: Peak DataFrame containing m/z, abundance, and other peak data
        """
        assert peak_df["mz"].is_sorted(), "Peak DataFrame must be sorted by m/z"
        assert (
            meta_df["ms_level"].n_unique() == 1
        ), "All spectra must be at the same MS level"

        self.peak_df = peak_df
        self.meta_df = meta_df
        self.frame_num_to_index = get_frame_num_to_index_arr(meta_df["frame_num"])

    @property
    def ms_level(self) -> int:
        """Get the MS level of this spectra group."""
        return self.meta_df["ms_level"].first()

    @property
    def num_frames(self) -> int:
        """Get the number of frames in this spectra group."""
        return self.meta_df.shape[0]

    def get_peak_container(self) -> PeakContainer:
        return PeakContainer(
            frame_num_arr=self.peak_df["frame_num"].to_numpy(),
            mz_arr=self.peak_df["mz"].to_numpy(),
            ab_arr=self.peak_df["ab"].to_numpy(),
            z_score_arr=self.peak_df["z_score"].to_numpy(),
        )

    def get_meta_container(self) -> MetaContainer:
        meta_df = self.meta_df
        return MetaContainer(
            frame_num_arr=meta_df["frame_num"].to_numpy(),
            frame_num_to_index_arr=self.frame_num_to_index,
            ms_level=meta_df.item(0, "ms_level"),
            rt_arr=meta_df["time_in_seconds"].to_numpy(),
            isolation_min_mz_arr=meta_df["isolation_min_mz"].to_numpy(),
            isolation_max_mz_arr=meta_df["isolation_max_mz"].to_numpy(),
        )

    def get_xic(self, mz: float, tolerance_in_ppm: float = 10) -> pl.DataFrame:
        """
        Extract ion chromatogram (XIC) for a specific m/z value.

        Args:
            mz: Target m/z value
            tolerance_in_ppm: Mass tolerance in parts per million

        Returns:
            DataFrame containing peaks within the specified m/z tolerance
        """
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
        """
        Get all peaks for a specific frame/spectrum.

        Args:
            frame_num: Frame number to extract

        Returns:
            DataFrame containing all peaks in the specified frame
        """
        frame_idx = self.frame_num_to_index(frame_num)
        return self.peak_df.filter(pl.col("frame_index") == frame_idx)

    def match(self, theoretical_peak_df, tolerance_in_ppm):
        """
        Match theoretical peaks against experimental peaks within tolerance.

        Args:
            theoretical_peak_df: DataFrame containing theoretical peak m/z values
            tolerance_in_ppm: Mass tolerance in parts per million

        Returns:
            DataFrame containing start/stop indices for matching peaks
        """
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
        """
        Create a peak DataFrame from arrays.

        Args:
            frame_num_arr: Array of frame numbers
            mz_arr: Array of m/z values
            ab_arr: Array of abundance/intensity values
            z_score_arr: Optional array of z-scores

        Returns:
            DataFrame sorted by m/z containing peak data
        """
        data = {
            "frame_num": frame_num_arr,
            "mz": mz_arr,
            "ab": ab_arr,
        }

        if z_score_arr is not None:
            data["z_score"] = z_score_arr

        return pl.DataFrame(data).sort(pl.col("mz"))
