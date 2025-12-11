from typing import Optional

import numpy as np
import polars as pl

from delpi.lcms.base_spectra import BaseSpectra


class MS1Spectra(BaseSpectra):
    """
    MS1 spectra group for handling MS1 level mass spectrometry data.

    This class manages MS1 spectra data including metadata and peak information,
    providing functionality for precursor feature detection and analysis.
    """

    def __init__(
        self,
        meta_df: pl.DataFrame,
        frame_num_arr: np.ndarray,
        mz_arr: np.ndarray,
        ab_arr: np.ndarray,
        z_score_arr: Optional[np.ndarray] = None,
    ):
        """
        Initialize MS1Spectra.

        Args:
            meta_df: Metadata DataFrame containing spectrum information
            frame_num_arr: Array of frame numbers
            mz_arr: Array of m/z values
            ab_arr: Array of abundance/intensity values
            z_score_arr: Optional array of z-scores for peak scoring
        """
        ms1_meta_df = meta_df.filter(pl.col("ms_level") == 1)
        peak_df = self.make_peak_df(frame_num_arr, mz_arr, ab_arr, z_score_arr)
        super().__init__(ms1_meta_df, peak_df)

        # rt_array = meta_df["time_in_seconds"].to_numpy()
        # self._rt_to_frame_index = interp1d(
        #     rt_array,
        #     np.arange(rt_array.shape[0]),
        #     bounds_error=False,
        #     fill_value="extrapolate",
        #     kind="nearest",
        # )

    # def get_aligend_peak_df(self, time_in_seconds: pl.Series):

    #     peak_df = self.peak_df.rename({"frame_index": "ms1_frame_index"})

    #     frame_indices = self._rt_to_frame_index(time_in_seconds)
    #     aligned_index_df = pl.DataFrame(
    #         frame_indices, schema={"ms1_frame_index": pl.UInt32}
    #     ).with_row_index(name="frame_index")

    #     aligned_peak_df = (
    #         peak_df.join(
    #             aligned_index_df,
    #             on="ms1_frame_index",
    #             how="inner",
    #         )
    #         .select(pl.exclude("ms1_frame_index"))
    #         .sort(pl.col("mz"))
    #     )

    #     return aligned_peak_df

    def find_precursor_features(self):
        """Find precursor features in the MS1 data."""
        raise NotImplementedError()
