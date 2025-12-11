from typing import List

import polars as pl
import numpy as np
import numba as nb

from delpi.model.rt_calibrator import RetentionTimeCalibrator


@nb.njit(cache=True)
def _kde_vote(
    grid: np.ndarray, rt: np.ndarray, w: np.ndarray, h: np.float32
) -> np.ndarray:
    inv2h2 = 1.0 / (2.0 * h * h)
    V = np.zeros(grid.size, dtype=np.float32)
    for g in range(grid.size):
        gg = grid[g]
        ssum = 0.0
        for i in range(rt.size):
            d = rt[i] - gg
            ssum += w[i] * np.exp(-(d * d) * inv2h2)
        V[g] = ssum
    return V


@nb.njit(cache=True)
def get_rt_with_kde(
    rt_arr: np.ndarray,
    weight_arr: np.ndarray,
    grid_step: np.float32 = 0.5,
    h: np.float32 = 3.0,
    W: np.float32 = -1.0,
) -> np.float32:

    if rt_arr.size == 1:
        return rt_arr[0]

    if rt_arr.size == 2:
        return rt_arr[np.argmax(weight_arr)]

    if W < 0:
        W = h
    # KDE → g*
    gmin = np.min(rt_arr) - 2.0 * h
    gmax = np.max(rt_arr) + 2.0 * h
    n_points = int((gmax - gmin) / grid_step) + 1
    grid = np.linspace(gmin, gmax, n_points).astype(np.float32)

    V = _kde_vote(grid, rt_arr, weight_arr, h)
    g_star = grid[int(np.argmax(V))]

    return g_star


@nb.njit(parallel=True, cache=True)
def batch_estimate_representative_retention_time(
    rt_arr: np.ndarray,
    weight_arr: np.ndarray,
    start_index_arr: np.ndarray,
    stop_index_arr: np.ndarray,
    xic_peak_interval: float = 1.5,
):
    n = len(start_index_arr)
    rep_rt_arr = np.empty(n, dtype=np.float32)
    for i in nb.prange(n):
        st = start_index_arr[i]
        ed = stop_index_arr[i]
        rep_rt_arr[i] = get_rt_with_kde(
            rt_arr=rt_arr[st:ed],
            weight_arr=weight_arr[st:ed],
            grid_step=xic_peak_interval * 0.5,
            h=xic_peak_interval,
        )
    return rep_rt_arr


def align_retention_times(
    pmsm_df: pl.DataFrame,
    rt_column: str = "observed_rt",
    aligned_rt_column: str = "aligned_rt",
    pmsm_sel_col: str = "apex_median_intensity",
) -> pl.DataFrame:

    cols = [
        "pmsm_index",
        "run_index",
        "precursor_index",
        "score",
    ]
    if pmsm_sel_col not in cols:
        cols.append(pmsm_sel_col)

    rt_df = pmsm_df.select(pl.col(*cols, rt_column))
    run_indices = rt_df["run_index"].unique().sort()

    if len(run_indices) <= 1:
        return rt_df.rename({rt_column: aligned_rt_column})

    ref_run_index = run_indices[0]
    ref_run_df = rt_df.filter(pl.col("run_index") == ref_run_index)

    aligned_dfs = list()
    aligned_dfs.append(ref_run_df.rename({rt_column: aligned_rt_column}))

    ref_rt_df = ref_run_df.group_by(["precursor_index"]).agg(
        pl.all().sort_by(pmsm_sel_col).last()
    )

    for run_index in run_indices[1:]:
        other_run_df = rt_df.filter(pl.col("run_index") == run_index)
        other_rt_df = other_run_df.group_by(["precursor_index"]).agg(
            pl.all().sort_by(pmsm_sel_col).last()
        )
        aligner = RetentionTimeCalibrator.train_aligner(
            ref_rt_df, other_rt_df, rt_column=rt_column
        )
        aligned_rt = aligner.predict(
            other_run_df[rt_column].to_numpy().reshape((-1, 1))
        ).flatten()
        aligned_dfs.append(
            other_run_df.select(pl.col(*cols)).with_columns(
                pl.Series(name=aligned_rt_column, values=aligned_rt, dtype=pl.Float32)
            )
        )

    return pl.concat(aligned_dfs)


def estimate_representative_retention_time(
    aligned_rt_df,
    weight_column="score",
    aligned_rt_column="aligned_rt",
    xic_peak_interval=1.5,
    rep_rt_column="representative_rt",
):

    if aligned_rt_df["run_index"].n_unique() == 1:
        return aligned_rt_df.group_by("precursor_index").agg(
            pl.col(aligned_rt_column).sort_by(weight_column).last().alias(rep_rt_column)
        )

    # aligned_rt_df = aligned_rt_df.sort(["precursor_index"])
    aligned_rt_df = aligned_rt_df.with_columns(
        (
            pl.col(weight_column)
            / pl.col(weight_column).sum().over(["run_index", "precursor_index"])
        ).alias("norm_weight")
    ).sort(["precursor_index"])

    n_rows = aligned_rt_df.shape[0]
    offset_df = (
        aligned_rt_df.with_row_index("_index")
        .unique("precursor_index", keep="first", maintain_order=True)
        .select(
            pl.col("precursor_index"),
            pl.col("_index").alias("start"),
            (pl.col("_index").shift(-1)).fill_null(n_rows).alias("stop"),
        )
    )

    start_index_arr = offset_df["start"].to_numpy()
    stop_index_arr = offset_df["stop"].to_numpy()

    rt_arr = aligned_rt_df[aligned_rt_column].to_numpy()
    weight_arr = aligned_rt_df["norm_weight"].to_numpy()

    rep_rt_arr = batch_estimate_representative_retention_time(
        rt_arr=rt_arr,
        weight_arr=weight_arr,
        start_index_arr=start_index_arr,
        stop_index_arr=stop_index_arr,
        xic_peak_interval=xic_peak_interval,
    )

    rep_rt_df = offset_df.select(
        pl.col("precursor_index"),
        pl.Series(name=rep_rt_column, values=rep_rt_arr),
    )

    return rep_rt_df
