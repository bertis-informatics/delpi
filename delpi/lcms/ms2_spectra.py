from typing import Optional

import numpy as np
import polars as pl

from delpi.lcms.base_spectra import BaseSpectra


class MS2Spectra(BaseSpectra):
    """
    MS2 spectra group for handling MS2 level mass spectrometry data.

    This class manages MS2 spectra data including metadata, peak information,
    and isolation window details for fragmentation analysis.
    """

    def __init__(
        self,
        group_index: int,
        meta_df: pl.DataFrame,
        frame_num_arr: np.ndarray,
        mz_arr: np.ndarray,
        ab_arr: np.ndarray,
        z_score_arr: Optional[np.ndarray] = None,
    ):
        """
        Initialize MS2Spectra.

        Args:
            group_index: Index identifier for this spectra group
            meta_df: Metadata DataFrame containing spectrum information
            frame_num_arr: Array of frame numbers
            mz_arr: Array of m/z values
            ab_arr: Array of abundance/intensity values
            z_score_arr: Optional array of z-scores for peak scoring
        """
        assert meta_df["frame_num"].is_sorted()
        self.group_index = group_index
        ms2_meta_df = meta_df.filter(pl.col("ms_level") == 2)
        peak_df = self.make_peak_df(frame_num_arr, mz_arr, ab_arr, z_score_arr)
        super().__init__(ms2_meta_df, peak_df)

        self.isolation_mz_range = ms2_meta_df.select(
            pl.col("isolation_min_mz").min(), pl.col("isolation_max_mz").max()
        ).row(0)
