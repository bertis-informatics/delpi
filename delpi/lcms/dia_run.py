import gc
from typing import Optional

import tqdm
import polars as pl

from pymsio import MassSpecData
from delpi.lcms.dia_window import DIAWindow
from delpi.lcms.ms1_spectra import MS1Spectra
from delpi.lcms.dia_scheme import determine_dia_scheme


class DIARun:

    def __init__(self, ms_data: MassSpecData):

        self.ms_data = ms_data
        self.name = ms_data.run_name
        self._init_meta()
        self._ms1_map = None
        self.windows = dict()

    def _init_meta(self):
        meta_df = self.ms_data.meta_df
        scheme_type, dia_scheme_df = determine_dia_scheme(meta_df)

        meta_df = meta_df.join(
            dia_scheme_df.select(pl.col("isolation_win_idx", "frame_num")).explode(
                "frame_num"
            ),
            on="frame_num",
            how="left",
        )
        self.meta_df = meta_df
        self.dia_scheme_df = dia_scheme_df
        self.scheme_type = scheme_type

    def load_windows(
        self,
        min_iso_win_idx: Optional[int] = None,
        max_iso_win_idx: Optional[int] = None,
        free_ms_data: bool = False,
    ):

        if min_iso_win_idx is None:
            min_iso_win_idx = self.dia_scheme_df.item(0, "isolation_win_idx")

        if max_iso_win_idx is None:
            max_iso_win_idx = self.dia_scheme_df.item(-1, "isolation_win_idx")

        pabar_total = max_iso_win_idx - min_iso_win_idx + 1 + 2
        with tqdm.tqdm(total=pabar_total, desc="Data-Prep") as pbar:

            self.ms_data.compute_z_score()
            pbar.update(1)

            # create MS1Spectra
            _ = self.get_ms1_map()
            pbar.update(1)

            for isolation_win_idx in range(min_iso_win_idx, max_iso_win_idx + 1):
                dia_window = self.get_dia_window(isolation_win_idx)
                self.windows[isolation_win_idx] = dia_window
                pbar.update(1)

        if free_ms_data:
            self.ms_data = None
            gc.collect()

    @property
    def gradient_length_in_seconds(self):
        if self.meta_df is None:
            raise RuntimeError("DIA data has not been loaded yet")
        return self.meta_df.item(-1, "time_in_seconds")

    @property
    def cycle_time_in_seconds(self):
        return float(
            self.meta_df.filter(pl.col("isolation_win_idx").is_not_null())
            .group_by("isolation_win_idx")
            .agg(
                (pl.col("time_in_seconds") - pl.col("time_in_seconds").shift(1)).mean()
            )["time_in_seconds"]
            .mean()
        )

    def get_ms1_map(self):

        if self._ms1_map is None:
            ms1_frame_nums = self.meta_df.filter(pl.col("ms_level") == 1)["frame_num"]
            frame_num_arr, mz_arr, ab_arr, z_arr = self.ms_data.collect_peaks(
                ms1_frame_nums
            )
            self._ms1_map = MS1Spectra(
                self.meta_df, frame_num_arr, mz_arr, ab_arr, z_arr
            )
        return self._ms1_map

    def get_dia_window(self, isolation_win_idx) -> Optional[DIAWindow]:

        if isolation_win_idx in self.windows:
            return self.windows[isolation_win_idx]

        ms2_frame_nums = self.meta_df.filter(
            pl.col("isolation_win_idx") == isolation_win_idx
        )["frame_num"]

        if ms2_frame_nums.shape[0] == 0:
            return None

        frame_num_arr, mz_arr, ab_arr, z_arr = self.ms_data.collect_peaks(
            ms2_frame_nums
        )

        win_df = self.dia_scheme_df.filter(
            pl.col("isolation_win_idx") == isolation_win_idx
        )
        isolation_min_mz = win_df.item(0, "isolation_min_mz")
        isolation_max_mz = win_df.item(0, "isolation_max_mz")
        dia_win = DIAWindow(
            isolation_win_idx,
            isolation_min_mz,
            isolation_max_mz,
            self.meta_df,
            frame_num_arr,
            mz_arr,
            ab_arr,
            z_arr,
        )

        return dia_win

    @classmethod
    def determine_dia_scheme(cls, meta_df: pl.DataFrame):
        scheme_type, dia_scheme_df = determine_dia_scheme(meta_df)
        return dia_scheme_df

    @classmethod
    def is_dia_run(cls, meta_df: pl.DataFrame):
        try:
            dia_scheme_df = cls.determine_dia_scheme(meta_df)
        except:
            return False

        return True
