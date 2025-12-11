import gc
from typing import Self

import numpy as np
import polars as pl
from scipy.signal import savgol_filter, find_peaks

from delpi.lcms.reader.base import MassSpecData
from delpi.lcms.ms1_spectra import MS1Spectra
from delpi.lcms.ms2_spectra import MS2Spectra
from delpi.lcms.data_container import SpectrumContainer


class DDARun:

    def __init__(self, ms_data: MassSpecData):

        self.ms_data = ms_data
        self.name = ms_data.run_name
        self.meta_df = ms_data.meta_df

        self.ms1_map = None
        self.ms2_maps = dict()

    @property
    def ms1_scan_interval(self) -> float:
        return (
            self.meta_df.filter(pl.col("ms_level") == 1)
            .select(pl.col("time_in_seconds").diff(1).median())
            .item()
        )

    @property
    def ms2_scan_interval(self) -> float:
        return (
            self.meta_df.filter(pl.col("ms_level") == 2)
            .select(pl.col("time_in_seconds").diff(1).median())
            .item()
        )

    @property
    def num_frames(self) -> int:
        return self.meta_df.shape[0]

    @property
    def isolation_mz_range(self):
        return (
            self.meta_df.filter(pl.col("ms_level") == 2)
            .select(pl.col("isolation_min_mz").min(), pl.col("isolation_max_mz").max())
            .row(0)
        )

    @property
    def gradient_length_in_seconds(self) -> float:
        if self.meta_df is None:
            raise RuntimeError("DIA data has not been loaded yet")
        return self.meta_df.item(-1, "time_in_seconds")

    def prepare(self, num_groups: int, free_ms_data: bool = False) -> Self:

        self.ms_data.compute_z_score()

        ms1_meta_df = self.meta_df.filter(pl.col("ms_level") == 1)
        ms2_meta_df = self.meta_df.filter(pl.col("ms_level") == 2)

        frame_num_arr, mz_arr, ab_arr, z_arr = self.ms_data.collect_peaks(
            ms1_meta_df["frame_num"]
        )
        self.ms1_map = MS1Spectra(self.meta_df, frame_num_arr, mz_arr, ab_arr, z_arr)

        if num_groups == 1:
            frame_num_arr, mz_arr, ab_arr, z_arr = self.ms_data.collect_peaks(
                ms2_meta_df["frame_num"]
            )
            ms2_map = MS2Spectra(0, self.meta_df, frame_num_arr, mz_arr, ab_arr, z_arr)
            self.ms2_maps[ms2_map.group_index] = ms2_map
        else:
            num_specs = int(np.round(ms2_meta_df.shape[0] / num_groups))
            for isolation_group_idx, sub_ms2_meta_df in enumerate(
                ms2_meta_df.sort("isolation_min_mz").iter_slices(num_specs)
            ):
                sub_ms2_meta_df = sub_ms2_meta_df.sort("frame_num")
                frame_num_arr, mz_arr, ab_arr, z_arr = self.ms_data.collect_peaks(
                    sub_ms2_meta_df["frame_num"]
                )
                ms2_map = MS2Spectra(
                    isolation_group_idx,
                    sub_ms2_meta_df,
                    frame_num_arr,
                    mz_arr,
                    ab_arr,
                    z_arr,
                )
                self.ms2_maps[ms2_map.group_index] = ms2_map

        if free_ms_data:
            self.ms_data.peak_arr = None
            self.ms_data.z_score_arr = None
            self.ms_data = None
            gc.collect()

        return self

    def get_spectrum(self, frame_num: int) -> SpectrumContainer:

        st, ed = self.ms_data.get_peak_index(frame_num)
        return SpectrumContainer(
            frame_num=frame_num,
            mz_arr=self.ms_data.peak_arr[st:ed, 0],
            ab_arr=self.ms_data.peak_arr[st:ed, 1],
            z_score_arr=self.ms_data.z_score_arr[st:ed],
        )

    def estimate_LC_fwhm(self, min_mz: float = 400, max_mz: float = 800):

        assert self.ms1_map is not None

        xic_arr = np.empty(self.ms1_map.meta_df.shape[0], dtype=np.float32)
        frame_num_to_index = self.ms1_map.frame_num_to_index

        peak_df = self.ms1_map.peak_df.filter(
            pl.col("mz").is_between(min_mz, max_mz)
        ).select(
            pl.col("mz").cast(pl.UInt16).alias("bin_idx"), pl.col("frame_num", "ab")
        )

        lc_peak_width_list = list()
        num_lc_peaks = 0
        for _, sub_df in peak_df.group_by(pl.col("bin_idx")):
            if sub_df.shape[0] > 32:
                xic_df = sub_df.group_by("frame_num").agg(pl.col("ab").sum())
                frame_indices = frame_num_to_index[xic_df["frame_num"]]

                xic_arr[:] = 0
                xic_arr[frame_indices] = xic_df["ab"]
                xic_arr = savgol_filter(xic_arr, window_length=11, polyorder=3)

                height_cutoff = xic_df["ab"].median() * 1.4
                peak_arr, peak_props = find_peaks(
                    xic_arr,
                    rel_height=0.5,
                    height=height_cutoff,
                    prominence=0.01,
                    width=5,
                )
                peak_st_arr = peak_props["left_ips"].astype(np.int32)
                peak_ed_arr = peak_props["right_ips"].astype(np.int32)
                peak_widths = (
                    self.ms1_map.meta_df[peak_ed_arr]["time_in_seconds"]
                    - self.ms1_map.meta_df[peak_st_arr]["time_in_seconds"]
                )
                lc_peak_width_list.append(peak_widths)
                num_lc_peaks += peak_arr.shape[0]
                # if num_lc_peaks > min_samples:
                #     break

        return float(pl.concat(lc_peak_width_list).median())
