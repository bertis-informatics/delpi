from typing import Optional

import numpy as np
import polars as pl
from scipy.interpolate import interp1d

from delpi.lcms.base_spectra import BaseSpectra
from delpi.lcms.data_container import DIAWindowFrameNumMap


class DIAWindow(BaseSpectra):

    def __init__(
        self,
        window_index: int,
        meta_df: pl.DataFrame,
        frame_num_arr: np.ndarray,
        mz_arr: np.ndarray,
        ab_arr: np.ndarray,
        z_score_arr: Optional[np.ndarray] = None,
    ):
        peak_df = self.make_peak_df(frame_num_arr, mz_arr, ab_arr, z_score_arr)

        ms1_meta_df = meta_df.filter(pl.col("ms_level") == 1)
        ms2_meta_df = meta_df.filter(pl.col("isolation_win_idx") == window_index)

        max_frame_num = meta_df.item(-1, "frame_num")
        frame_num_to_index = np.full(max_frame_num + 1, -1, dtype=np.int32)

        ## create frame_num_to_index mapping
        # mapping MS2 frame_nums
        for frame_index, frame_num in enumerate(ms2_meta_df["frame_num"]):
            frame_num_to_index[frame_num] = frame_index

        # mapping MS1 frame_nums
        ms1_rt_array = ms1_meta_df["time_in_seconds"].to_numpy()
        ms1_frame_indices = interp1d(
            ms1_rt_array,
            np.arange(ms1_rt_array.shape[0]),
            bounds_error=False,
            fill_value="extrapolate",
            kind="nearest",
        )(ms2_meta_df["time_in_seconds"]).astype(np.uint32)

        ms1_meta_df = ms1_meta_df[ms1_frame_indices]
        for frame_index, frame_num in enumerate(ms1_meta_df["frame_num"]):
            frame_num_to_index[frame_num] = frame_index

        assert ms1_meta_df.shape[0] == ms2_meta_df.shape[0]

        super().__init__(ms2_meta_df, peak_df)

        self.window_index = window_index
        self.ms1_meta_df = ms1_meta_df
        self.frame_num_to_index = frame_num_to_index

    @property
    def isolation_mz_range(self):
        return self.meta_df.item(0, "isolation_min_mz"), self.meta_df.item(
            0, "isolation_max_mz"
        )

    def get_frame_num_map(self):
        return DIAWindowFrameNumMap(
            self.ms1_meta_df["frame_num"].to_numpy(),
            self.meta_df["frame_num"].to_numpy(),
            self.ms1_meta_df["time_in_seconds"].to_numpy(),
            self.meta_df["time_in_seconds"].to_numpy(),
            self.frame_num_to_index,
        )
