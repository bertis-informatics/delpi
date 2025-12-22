import gc
from typing import Optional

import tqdm
import numpy as np
import polars as pl

from delpi.lcms.reader.base import MassSpecData
from delpi.lcms.dia_window import DIAWindow
from delpi.lcms.ms1_spectra import MS1Spectra


class DIARun:

    def __init__(self, ms_data: MassSpecData):

        self.ms_data = ms_data
        self.name = ms_data.run_name
        self._init_meta()
        self._ms1_map = None
        self.windows = dict()

    def _init_meta(self):
        meta_df = self.ms_data.meta_df
        dia_scheme_df = self.determine_dia_scheme(meta_df)
        # num_spectra_per_window = dia_scheme_df.select(
        #     pl.col("frame_num").list.len()
        # ).unique()
        # assert num_spectra_per_window.shape[0] == 1
        meta_df = meta_df.join(
            dia_scheme_df.select(pl.col("isolation_win_idx", "frame_num")).explode(
                "frame_num"
            ),
            on="frame_num",
            how="left",
        )
        self.meta_df = meta_df
        self.dia_scheme_df = dia_scheme_df

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
            self.ms_data.peak_arr = None
            self.ms_data.z_score_arr = None
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

        dia_win = DIAWindow(
            isolation_win_idx, self.meta_df, frame_num_arr, mz_arr, ab_arr, z_arr
        )

        return dia_win

    @classmethod
    def determine_dia_scheme(cls, meta_df: pl.DataFrame):
        # [TODO] more sophisticated implementation needed for handling staggered or overlapped windows
        ms2_win_df = (
            meta_df.filter(pl.col("ms_level") == 2)
            .select(pl.col("frame_num", "isolation_min_mz", "isolation_max_mz"))
            .with_columns(
                pl.col("isolation_min_mz").cast(pl.Float64).round(1).alias("win_lb"),
                pl.col("isolation_max_mz").cast(pl.Float64).round(1).alias("win_ub"),
            )
        )
        iso_win_df = (
            ms2_win_df.group_by(["win_lb", "win_ub"])
            .agg(
                pl.col("isolation_min_mz").min(),
                pl.col("isolation_max_mz").max(),
                pl.len().alias("num_frames"),
            )
            .sort("win_lb")
        ).with_row_index("index")

        if iso_win_df["num_frames"].n_unique() > 5:
            raise ValueError(
                "The number of spectra per regular window significantly varies"
            )

        ms2_win_df = ms2_win_df.join(
            iso_win_df.select(pl.col("index", "win_lb", "win_ub")),
            on=["win_lb", "win_ub"],
            how="left",
        ).drop(["win_lb", "win_ub"])
        iso_win_df = iso_win_df.drop(["win_lb", "win_ub"])

        get_win_width = (pl.col("isolation_max_mz") - pl.col("isolation_min_mz")).alias(
            "win_width"
        )
        get_overlapped = (
            pl.col("isolation_max_mz").shift(1) - pl.col("isolation_min_mz")
        ).alias("overlapped")
        iso_win_df = iso_win_df.with_columns(get_win_width, get_overlapped)
        is_overlapped, is_staggered = (
            iso_win_df[1:-1]
            .select(
                is_overlapped=(
                    (pl.col("overlapped") > 0.3)
                    & (
                        (pl.col("overlapped") < pl.col("win_width") * 0.3)
                        | (pl.col("overlapped") < 1.5)
                    )
                ),
                is_staggered=(pl.col("overlapped") > pl.col("win_width") * 0.4).all(),
            )
            .row(0)
        )

        if is_staggered:
            idx, min_mz, max_mz = iso_win_df.select(
                pl.col("index", "isolation_min_mz", "isolation_max_mz")
            ).row(0)
            staggered_iso_wins = [[[idx], min_mz, max_mz]]
            prev_max_mz = max_mz

            for i in range(1, iso_win_df.shape[0] - 1):
                staggered_win_df = iso_win_df[i : i + 2]
                min_mz, max_mz = prev_max_mz, staggered_win_df.item(
                    0, "isolation_max_mz"
                )
                staggered_iso_wins.append(
                    [staggered_win_df["index"].to_list(), min_mz, max_mz]
                )
                prev_max_mz = max_mz

            idx = iso_win_df.item(-1, "index")
            min_mz, max_mz = prev_max_mz, iso_win_df.item(-1, "isolation_max_mz")
            staggered_iso_wins.append([[idx], min_mz, max_mz])
            iso_win_df = pl.DataFrame(
                staggered_iso_wins,
                schema={
                    "index": pl.List(pl.UInt32),
                    "isolation_min_mz": pl.Float32,
                    "isolation_max_mz": pl.Float32,
                },
                orient="row",
            ).with_row_index("isolation_win_idx")

            dia_scheme_df = (
                iso_win_df.explode("index")
                .join(
                    ms2_win_df.select(pl.col("index", "frame_num")),
                    on="index",
                    how="left",
                )
                .group_by("isolation_win_idx")
                .agg(
                    pl.col("isolation_min_mz").first(),
                    pl.col("isolation_max_mz").first(),
                    pl.col("frame_num"),
                )
            )
        else:

            if is_overlapped:
                overlapped_mz = iso_win_df["overlapped"].to_numpy()
                min_mz = iso_win_df["isolation_min_mz"].to_numpy().copy()
                max_mz = iso_win_df["isolation_max_mz"].to_numpy().copy()

                max_mz[:-1] = max_mz[:-1] - overlapped_mz[1:] * 0.5
                min_mz[1:] = min_mz[1:] + overlapped_mz[1:] * 0.5

                iso_win_df = iso_win_df.select(pl.col("index")).with_columns(
                    isolation_min_mz=min_mz, isolation_max_mz=max_mz
                )
            else:
                overlapped_mz = iso_win_df["overlapped"].to_numpy()
                if np.any(overlapped_mz[1:] > 0.1):
                    raise ValueError("Cannot determine DIA scheme")

                iso_win_df = iso_win_df.select(
                    pl.col("index", "isolation_min_mz", "isolation_max_mz")
                )

            dia_scheme_df = (
                iso_win_df.join(
                    ms2_win_df.select(pl.col("index", "frame_num")),
                    on="index",
                    how="left",
                )
                .group_by("index")
                .agg(
                    pl.col("isolation_min_mz").first(),
                    pl.col("isolation_max_mz").first(),
                    pl.col("frame_num"),
                )
            ).rename({"index": "isolation_win_idx"})

        return dia_scheme_df

    @classmethod
    def is_dia_run(cls, meta_df: pl.DataFrame):
        try:
            dia_scheme_df = cls.determine_dia_scheme(meta_df)
        except:
            return False

        return True


def get_test_meta_df(staggered=False, overlapped=False):

    win_width = 8 if staggered else 4
    if not staggered and overlapped:
        win_width += 1

    i = 0
    meta_list = []
    for _ in range(100):
        for st in range(300, 900, 4):
            meta_list.append([i, st, st + win_width])
            i += 1
    meta_df = pl.DataFrame(
        meta_list,
        orient="row",
        schema={
            "frame_num": pl.UInt32,
            "isolation_min_mz": pl.Float32,
            "isolation_max_mz": pl.Float32,
        },
    ).with_columns(ms_level=pl.lit(2))
    return meta_df
