"""Peak-explanation NMS for DDA PmSM selection after rescoring.

Per run, PmSMs are visited by descending score (ties: ascending
``pmsm_index``). A PmSM is suppressed when at least ``threshold`` of its
center-spectrum M/M+1 peaks are already claimed by accepted PmSMs; otherwise
it is accepted and claims its peaks. A precursor can be accepted in several
spectra; FDR uses its best accepted PmSM per run.

Peak indices refer to the peak table of one isolation group, so they are
keyed by (search_group, peak index). At search time a pairwise variant over
the pre-trained logits defines the stored clusters (`peak_nms_clusters`): a
PmSM joins a head only if it shares >= threshold of its peaks with that head.
"""

import logging

import numba as nb
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_EXPLAINED_FRACTION = 0.5

# per-PmSM outcome of the NMS
ACCEPTED = 1
EXPLAINED = 2  # suppressed: center peaks already explained

_FRACTION_EPS = 1e-9


@nb.njit(cache=True)
def _spectrum_peak_nms(
    order: np.ndarray,
    peak_ptr: np.ndarray,
    peak_id: np.ndarray,
    n_peak_ids: int,
    threshold: float,
) -> np.ndarray:
    """Greedy NMS over one run; `peak_ptr`/`peak_id` are each row's dense
    peak ids (CSR). Returns ACCEPTED/EXPLAINED per row."""
    n = order.shape[0]
    status = np.zeros(n, dtype=np.int8)
    claimed = np.zeros(n_peak_ids, dtype=np.bool_)

    for o in range(n):
        i = order[o]
        s, e = peak_ptr[i], peak_ptr[i + 1]
        if e > s:
            explained = 0
            for k in range(s, e):
                if claimed[peak_id[k]]:
                    explained += 1
            if explained / (e - s) >= threshold - _FRACTION_EPS:
                status[i] = EXPLAINED
                continue
        status[i] = ACCEPTED
        for k in range(s, e):
            claimed[peak_id[k]] = True

    return status


@nb.njit(cache=True)
def _head_clusters(
    order: np.ndarray,
    peak_ptr: np.ndarray,
    peak_id: np.ndarray,
    n_peak_ids: int,
    threshold: float,
) -> np.ndarray:
    """Cluster id per row. Rows are visited in `order`; an unassigned row
    becomes a head, and every unassigned row sharing >= `threshold` of its own
    peaks with that head joins the head's cluster."""
    n = order.shape[0]
    # rows of each peak id (CSR)
    peak_rows_ptr = np.zeros(n_peak_ids + 1, dtype=np.int64)
    for k in range(peak_id.shape[0]):
        peak_rows_ptr[peak_id[k] + 1] += 1
    peak_rows_ptr = np.cumsum(peak_rows_ptr)
    peak_rows = np.empty(peak_id.shape[0], dtype=np.int64)
    fill = peak_rows_ptr[:-1].copy()
    for i in range(n):
        for k in range(peak_ptr[i], peak_ptr[i + 1]):
            peak_rows[fill[peak_id[k]]] = i
            fill[peak_id[k]] += 1

    cluster = np.full(n, -1, dtype=np.int64)
    shared = np.zeros(n, dtype=np.int64)
    touched = np.empty(n, dtype=np.int64)
    n_clusters = 0
    for o in range(n):
        h = order[o]
        if cluster[h] >= 0:
            continue
        cluster[h] = n_clusters
        n_touched = 0
        for k in range(peak_ptr[h], peak_ptr[h + 1]):
            p = peak_id[k]
            for t in range(peak_rows_ptr[p], peak_rows_ptr[p + 1]):
                j = peak_rows[t]
                if cluster[j] >= 0:
                    continue
                if shared[j] == 0:
                    touched[n_touched] = j
                    n_touched += 1
                shared[j] += 1
        for t in range(n_touched):
            j = touched[t]
            if shared[j] / (peak_ptr[j + 1] - peak_ptr[j]) >= threshold - _FRACTION_EPS:
                cluster[j] = n_clusters
            shared[j] = 0
        n_clusters += 1
    return cluster


def _dense_peak_ids(peaks: np.ndarray, search_group: np.ndarray):
    """CSR (ptr, ids) of each row's unique (search_group, peak) keys as dense ids."""
    rows, cols = np.nonzero(peaks >= 0)
    keys = (search_group[rows].astype(np.int64) << 32) | peaks[rows, cols].astype(
        np.int64
    )
    # a peak matched by two fragments counts once
    pairs = np.unique(np.stack([rows.astype(np.int64), keys], axis=1), axis=0)
    unique_keys, ids = np.unique(pairs[:, 1], return_inverse=True)
    counts = np.bincount(pairs[:, 0], minlength=peaks.shape[0])
    ptr = np.zeros(peaks.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr, ids.astype(np.int64), unique_keys.shape[0]


def peak_nms_status(
    score: np.ndarray,
    pmsm_index: np.ndarray,
    center_peaks: np.ndarray,
    search_group: np.ndarray,
    threshold: float = DEFAULT_EXPLAINED_FRACTION,
) -> np.ndarray:
    """Peak NMS for one run. `center_peaks` is ``(N, W)``, -1 padded.
    Returns ACCEPTED/EXPLAINED per row."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if score.shape[0] == 0:
        return np.zeros(0, dtype=np.int8)
    if not np.isfinite(score).all():
        raise ValueError("PmSM scores must be finite")

    order = np.lexsort((pmsm_index, -score.astype(np.float64)))
    peak_ptr, peak_id, n_peak_ids = _dense_peak_ids(center_peaks, search_group)
    return _spectrum_peak_nms(
        order.astype(np.int64), peak_ptr, peak_id, n_peak_ids, float(threshold)
    )


def peak_nms_clusters(
    score: np.ndarray,
    center_peaks: np.ndarray,
    threshold: float = DEFAULT_EXPLAINED_FRACTION,
) -> np.ndarray:
    """Cluster ids (dense from 0) of PmSMs whose center peaks index one peak
    table. In descending `score`, each PmSM not yet in a cluster becomes a
    head, and the remaining PmSMs sharing >= `threshold` of their own center
    peaks with that head join it (pairwise NMS)."""
    n = score.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    if not np.isfinite(score).all():
        raise ValueError("PmSM scores must be finite")
    order = np.lexsort((np.arange(n), -score.astype(np.float64)))
    peak_ptr, peak_id, n_peak_ids = _dense_peak_ids(
        center_peaks, np.zeros(n, dtype=np.uint32)
    )
    return _head_clusters(
        order.astype(np.int64), peak_ptr, peak_id, n_peak_ids, float(threshold)
    ).astype(np.int32)


def best_accepted_per_precursor(run_df: pl.DataFrame, status: np.ndarray) -> pl.DataFrame:
    """Each precursor's best accepted PmSM (score desc, then pmsm_index asc)."""
    return (
        run_df.filter(pl.Series(status == ACCEPTED))
        .sort(["score", "pmsm_index"], descending=[True, False])
        .unique("precursor_index", keep="first", maintain_order=True)
    )


def select_pmsms_by_peak_nms(
    scored_pmsm_df: pl.DataFrame,
    result_aggregator,
    group_key: str,
    threshold: float = DEFAULT_EXPLAINED_FRACTION,
) -> pl.DataFrame:
    """Each precursor's best accepted PmSM per run, with
    ``n_accepted_spectra`` (its accepted PmSMs in the run). Center peaks are
    read from the result files."""
    selected_dfs = []
    totals = {"pmsm": 0, "accepted": 0, "explained": 0, "precursors": 0}
    for (run_index,), run_df in scored_pmsm_df.group_by(
        "run_index", maintain_order=True
    ):
        result_manager = result_aggregator.get_result_manager(run_index)
        stored = result_manager.read_dict(
            group_key, data_keys=["center_peak_index", "search_group"]
        )
        if "center_peak_index" not in stored:
            raise ValueError(
                f"{result_manager.hdf_file_path}: no center_peak_index in "
                f"{group_key}; re-run the DDA search with this DelPi version"
            )
        rows = run_df["pmsm_index"].to_numpy()
        status = peak_nms_status(
            run_df["score"].to_numpy(),
            rows,
            stored["center_peak_index"][rows],
            stored["search_group"][rows],
            threshold,
        )
        n_accepted = (
            run_df.filter(pl.Series(status == ACCEPTED))
            .group_by("precursor_index")
            .len("n_accepted_spectra")
        )
        best = best_accepted_per_precursor(run_df, status).join(
            n_accepted, on="precursor_index", how="left"
        )
        selected_dfs.append(best)
        totals["pmsm"] += run_df.height
        totals["accepted"] += int(np.count_nonzero(status == ACCEPTED))
        totals["explained"] += int(np.count_nonzero(status == EXPLAINED))
        totals["precursors"] += best.height

    logger.info(
        "Peak NMS (threshold %g): %s PmSMs, %s accepted, %s suppressed; "
        "%s precursors to FDR",
        threshold,
        f"{totals['pmsm']:,}",
        f"{totals['accepted']:,}",
        f"{totals['explained']:,}",
        f"{totals['precursors']:,}",
    )
    if not selected_dfs:
        return scored_pmsm_df.clear().with_columns(
            pl.lit(None, dtype=pl.UInt32).alias("n_accepted_spectra")
        )
    return pl.concat(selected_dfs, how="vertical")
