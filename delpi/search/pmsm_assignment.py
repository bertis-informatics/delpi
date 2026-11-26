"""One-run PmSM assignment by an order-preserving DP.

PmSM ``cluster`` is the run-local fragment-peak-sharing competition group
defined upstream. Within each alignment group (connected component of the
precursor-cluster graph), the DP picks a one-to-one precursor-cluster
assignment and one PmSM per pair, preserving predicted-RT order; nothing is
preselected. Groups with one precursor take the PmSM with the highest
median_intensity. Otherwise the utility is score plus log2 intensity relative
to the precursor's strongest candidate, so absolute abundance never favours one
precursor over another. Runs before FDR control; is_decoy is not used.
"""

import numba as nb
import numpy as np
import polars as pl

REQUIRED_COLUMNS = (
    "precursor_index",
    "frame_num",
    "cluster",
    "predicted_rt",
    "pmsm_index",
    "score",
    "median_intensity",
)

# tolerance for comparing DP utility / raw-score totals
_TIE_TOLERANCE = 1e-8
# stabilizes the precursor-relative log2 intensity ratio when intensities are 0
_INTENSITY_EPSILON = 1e-6


# union-find


@nb.njit(cache=True)
def _uf_find(parent: np.ndarray, x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


@nb.njit(cache=True)
def _uf_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1


@nb.njit(cache=True)
def _union_find_components(
    precursor_node: np.ndarray,
    cluster_node: np.ndarray,
    n_precursor_nodes: int,
    n_cluster_nodes: int,
) -> np.ndarray:
    """Connected components of the precursor-cluster graph.

    Precursor nodes are ``[0, n_precursor_nodes)``, cluster nodes follow. Each
    PmSM row is an edge. Returns the (non-dense) root id of every node.
    """
    n = n_precursor_nodes + n_cluster_nodes
    parent = np.arange(n).astype(np.uint32)
    rank = np.zeros(n, dtype=np.uint32)
    for i in range(precursor_node.shape[0]):
        _uf_union(parent, rank, precursor_node[i], n_precursor_nodes + cluster_node[i])
    component = np.empty(n, dtype=np.uint32)
    for i in range(n):
        component[i] = _uf_find(parent, i)
    return component


@nb.njit(cache=True)
def _unique_inverse(values: np.ndarray) -> tuple:
    """``np.unique(values, return_inverse=True)``, which numba does not support."""
    sorted_values = np.sort(values)
    unique_values = np.empty(sorted_values.shape[0], dtype=values.dtype)
    cnt = 0
    for i in range(sorted_values.shape[0]):
        if cnt == 0 or sorted_values[i] != unique_values[cnt - 1]:
            unique_values[cnt] = sorted_values[i]
            cnt += 1
    unique_values = unique_values[:cnt]
    inverse = np.searchsorted(unique_values, values).astype(np.int32)
    return unique_values, inverse


# order-preserving assignment DP


@nb.njit(cache=True)
def _is_strictly_better(
    utility_a: float,
    score_a: float,
    count_a: int,
    utility_b: float,
    score_b: float,
    count_b: int,
    tol: float,
) -> bool:
    """Whether DP state A beats B by utility, then raw score (within `tol`),
    then count. False on a full tie, so the caller's evaluation order decides.
    """
    # if count_a != count_b:
    #     return count_a > count_b
    # if utility_a > utility_b + tol:
    #     return True
    # if utility_a < utility_b - tol:
    #     return False
    # if score_a > score_b + tol:
    #     return True
    # if score_a < score_b - tol:
    #     return False
    # return False

    if utility_a > utility_b + tol:
        return True
    if utility_a < utility_b - tol:
        return False

    if score_a > score_b + tol:
        return True
    if score_a < score_b - tol:
        return False

    return count_a > count_b


@nb.njit(cache=True)
def _dp_assign(
    utility: np.ndarray, raw_score: np.ndarray, valid: np.ndarray, tol: float
) -> np.ndarray:
    """Order-preserving assignment DP for one alignment group.

    Inputs are ``(M, N)`` over precursors (rows, ``predicted_rt`` order) and
    clusters (cols, representative ``frame_num`` order); `valid[i, j]` marks a
    PmSM linking them; a skip scores 0. Maximizes total utility, then raw
    score, then match count; exact ties prefer match, then precursor skip,
    then cluster skip.

    Returns a ``(k, 2)`` array of ``(precursor_row, cluster_col)`` matches.
    """
    m, n = utility.shape
    dp_utility = np.zeros((m + 1, n + 1), dtype=np.float64)
    dp_score = np.zeros((m + 1, n + 1), dtype=np.float64)
    dp_count = np.zeros((m + 1, n + 1), dtype=np.int32)
    # 0 = precursor skip, 1 = cluster skip, 2 = matched
    choice = np.zeros((m + 1, n + 1), dtype=np.int8)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            has_best = False
            best_utility = 0.0
            best_score = 0.0
            best_count = 0
            best_choice = 0

            if valid[i - 1, j - 1]:
                best_utility = dp_utility[i - 1, j - 1] + utility[i - 1, j - 1]
                best_score = dp_score[i - 1, j - 1] + raw_score[i - 1, j - 1]
                best_count = dp_count[i - 1, j - 1] + 1
                best_choice = 2
                has_best = True

            cand_utility, cand_score, cand_count = (
                dp_utility[i - 1, j],
                dp_score[i - 1, j],
                dp_count[i - 1, j],
            )
            if not has_best:
                best_utility, best_score, best_count, best_choice = (
                    cand_utility,
                    cand_score,
                    cand_count,
                    0,
                )
                has_best = True
            elif _is_strictly_better(
                cand_utility,
                cand_score,
                cand_count,
                best_utility,
                best_score,
                best_count,
                tol,
            ):
                best_utility, best_score, best_count, best_choice = (
                    cand_utility,
                    cand_score,
                    cand_count,
                    0,
                )

            cand_utility, cand_score, cand_count = (
                dp_utility[i, j - 1],
                dp_score[i, j - 1],
                dp_count[i, j - 1],
            )
            if _is_strictly_better(
                cand_utility,
                cand_score,
                cand_count,
                best_utility,
                best_score,
                best_count,
                tol,
            ):
                best_utility, best_score, best_count, best_choice = (
                    cand_utility,
                    cand_score,
                    cand_count,
                    1,
                )

            dp_utility[i, j] = best_utility
            dp_score[i, j] = best_score
            dp_count[i, j] = best_count
            choice[i, j] = best_choice

    matches = np.empty((min(m, n), 2), dtype=np.int32)
    cnt = 0
    i, j = m, n
    while i > 0 and j > 0:
        c = choice[i, j]
        if c == 2:
            matches[cnt, 0] = i - 1
            matches[cnt, 1] = j - 1
            cnt += 1
            i -= 1
            j -= 1
        elif c == 0:
            i -= 1
        else:
            j -= 1
    return matches[:cnt][::-1]


@nb.njit(cache=True)
def _dp_assign_groups(
    group_start: np.ndarray,
    group_end: np.ndarray,
    precursor_order: np.ndarray,
    cluster_order: np.ndarray,
    assignment_utility: np.ndarray,
    raw_score: np.ndarray,
    pmsm_index: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Run `_dp_assign` per alignment group over flat, group-sorted arrays.

    `group_start`/`group_end` delimit each group's rows. When several PmSMs
    link the same precursor and cluster, the cell keeps the best
    utility/score, then the smallest `pmsm_index`. Returns the selected
    `pmsm_index` values of all groups.
    """
    out = np.empty(precursor_order.shape[0], dtype=pmsm_index.dtype)
    out_cnt = 0
    for g in range(group_start.shape[0]):
        s, e = group_start[g], group_end[g]
        p_uniq, p_local = _unique_inverse(precursor_order[s:e])
        c_uniq, c_local = _unique_inverse(cluster_order[s:e])
        m, n = p_uniq.shape[0], c_uniq.shape[0]

        utility = np.zeros((m, n), dtype=np.float64)
        raw = np.zeros((m, n), dtype=np.float64)
        valid = np.zeros((m, n), dtype=np.bool_)
        # only `valid` cells are read
        pmsm_grid = np.zeros((m, n), dtype=pmsm_index.dtype)
        for k in range(s, e):
            i, j = p_local[k - s], c_local[k - s]
            if valid[i, j]:
                better = _is_strictly_better(
                    assignment_utility[k],
                    raw_score[k],
                    1,
                    utility[i, j],
                    raw[i, j],
                    1,
                    tol,
                )
                tied = (
                    abs(assignment_utility[k] - utility[i, j]) <= tol
                    and abs(raw_score[k] - raw[i, j]) <= tol
                )
                if not (better or (tied and pmsm_index[k] < pmsm_grid[i, j])):
                    continue
            utility[i, j] = assignment_utility[k]
            raw[i, j] = raw_score[k]
            valid[i, j] = True
            pmsm_grid[i, j] = pmsm_index[k]

        for i, j in _dp_assign(utility, raw, valid, tol):
            out[out_cnt] = pmsm_grid[i, j]
            out_cnt += 1
    return out[:out_cnt]


# Polars-level pipeline


def _compute_intensity_utilities(
    pmsm_df: pl.DataFrame, intensity_weight: float
) -> pl.DataFrame:
    """Add ``_relative_log_intensity`` and ``_assignment_utility`` columns.

    ``_relative_log_intensity`` is log2 of ``median_intensity`` over the
    maximum for the same ``precursor_index`` (input is one run): <= 0, and 0
    for the strongest candidate.
    """
    max_intensity = pl.col("median_intensity").max().over("precursor_index")
    relative_log_intensity = (
        (pl.col("median_intensity") + _INTENSITY_EPSILON)
        / (max_intensity + _INTENSITY_EPSILON)
    ).log(base=2)

    df = pmsm_df.with_columns(relative_log_intensity.alias("_relative_log_intensity"))
    return df.with_columns(
        (pl.col("score") + intensity_weight * pl.col("_relative_log_intensity")).alias(
            "_assignment_utility"
        )
    )


def _find_alignment_groups(run_df: pl.DataFrame) -> pl.DataFrame:
    """Add ``_alignment_group``: the (non-dense) connected component id of the
    run's precursor_index-cluster graph. Not the same as `cluster`.
    """
    precursor_keys = (
        run_df.select("precursor_index")
        .unique()
        .sort("precursor_index")
        .with_row_index("_precursor_node")
    )
    cluster_keys = (
        run_df.select("cluster")
        .unique()
        .sort("cluster")
        .with_row_index("_cluster_node")
    )
    node_df = run_df.join(precursor_keys, on="precursor_index", how="left").join(
        cluster_keys, on="cluster", how="left"
    )
    n_precursor_nodes = precursor_keys.height
    n_cluster_nodes = cluster_keys.height
    component = _union_find_components(
        node_df["_precursor_node"].to_numpy().astype(np.int64),
        node_df["_cluster_node"].to_numpy().astype(np.int64),
        n_precursor_nodes,
        n_cluster_nodes,
    )
    precursor_keys = precursor_keys.with_columns(
        pl.Series("_alignment_group", component[:n_precursor_nodes], dtype=pl.UInt32)
    )
    return run_df.join(
        precursor_keys.select(["precursor_index", "_alignment_group"]),
        on="precursor_index",
        how="left",
    )


def _prepare_assignment_problem(run_df: pl.DataFrame) -> pl.DataFrame:
    """DP input: one row per PmSM with `alignment_group`, `precursor_order`
    (``predicted_rt``, then `precursor_index`) and `cluster_order` (median
    `frame_num`, then `cluster`), ranked within the group.
    """
    precursor_order_df = (
        run_df.select(["_alignment_group", "precursor_index", "predicted_rt"])
        .unique()
        .sort(["_alignment_group", "predicted_rt", "precursor_index"])
        .with_columns(
            (pl.cum_count("precursor_index").over("_alignment_group") - 1).alias(
                "precursor_order"
            )
        )
        .select(["_alignment_group", "precursor_index", "precursor_order"])
    )
    cluster_order_df = (
        run_df.group_by(["_alignment_group", "cluster"])
        .agg(pl.col("frame_num").median().alias("_representative_frame"))
        .sort(["_alignment_group", "_representative_frame", "cluster"])
        .with_columns(
            (pl.cum_count("cluster").over("_alignment_group") - 1).alias(
                "cluster_order"
            )
        )
        .select(["_alignment_group", "cluster", "cluster_order"])
    )
    return (
        run_df.select(
            [
                "precursor_index",
                "cluster",
                "pmsm_index",
                "score",
                "median_intensity",
                "_relative_log_intensity",
                "_assignment_utility",
                "_alignment_group",
            ]
        )
        .join(
            precursor_order_df, on=["_alignment_group", "precursor_index"], how="inner"
        )
        .join(cluster_order_df, on=["_alignment_group", "cluster"], how="inner")
        .rename({"_alignment_group": "alignment_group"})
        .sort(["alignment_group", "precursor_order", "cluster_order"])
    )


def _solve_monotonic_assignment(
    alignment_group: np.ndarray,
    precursor_order: np.ndarray,
    cluster_order: np.ndarray,
    assignment_utility: np.ndarray,
    raw_score: np.ndarray,
    pmsm_index: np.ndarray,
) -> np.ndarray:
    """Solve the DP per alignment group and return the selected `pmsm_index`.

    `alignment_group` must be sorted so each group's rows are contiguous.
    """
    group_boundary = np.flatnonzero(np.diff(alignment_group)) + 1
    group_start = np.concatenate(([0], group_boundary)).astype(np.int64)
    group_end = np.concatenate((group_boundary, [alignment_group.shape[0]])).astype(
        np.int64
    )
    return _dp_assign_groups(
        group_start,
        group_end,
        precursor_order.astype(np.int32),
        cluster_order.astype(np.int32),
        assignment_utility.astype(np.float64),
        raw_score.astype(np.float64),
        pmsm_index,
        _TIE_TOLERANCE,
    )


def _select_single_precursor_groups(edge_candidates: pl.DataFrame) -> np.ndarray:
    """Single-precursor groups: pick the highest `median_intensity` PmSM
    (ties: higher `score`, then lower `pmsm_index`).
    """
    best = (
        edge_candidates.sort(
            ["alignment_group", "median_intensity", "score", "pmsm_index"],
            descending=[False, True, True, False],
        )
        .group_by("alignment_group", maintain_order=True)
        .first()
    )
    return best["pmsm_index"].to_numpy()


def _assign_run(run_df: pl.DataFrame) -> np.ndarray:
    """Return the selected `pmsm_index` values for one run.

    Single-precursor groups pick by intensity; the rest run the DP over
    `_assignment_utility` (score + intensity_weight * relative log intensity).
    """
    run_df = _find_alignment_groups(run_df)
    edge_candidates = _prepare_assignment_problem(run_df)
    if edge_candidates.height == 0:
        return np.empty(0, dtype=run_df["pmsm_index"].to_numpy().dtype)

    precursor_count = edge_candidates.group_by("alignment_group").agg(
        pl.col("precursor_index").n_unique().alias("_n_precursors")
    )
    edge_candidates = edge_candidates.join(
        precursor_count, on="alignment_group", how="left"
    )

    single_precursor = edge_candidates.filter(pl.col("_n_precursors") == 1)
    multi_precursor = edge_candidates.filter(pl.col("_n_precursors") > 1)

    selected = []
    if single_precursor.height > 0:
        selected.append(_select_single_precursor_groups(single_precursor))
    if multi_precursor.height > 0:
        multi_precursor = multi_precursor.sort(
            ["alignment_group", "precursor_order", "cluster_order"]
        )
        selected.append(
            _solve_monotonic_assignment(
                multi_precursor["alignment_group"].to_numpy(),
                multi_precursor["precursor_order"].to_numpy(),
                multi_precursor["cluster_order"].to_numpy(),
                multi_precursor["_assignment_utility"].to_numpy(),
                multi_precursor["score"].to_numpy(),
                multi_precursor["pmsm_index"].to_numpy(),
            )
        )

    if not selected:
        return np.empty(0, dtype=run_df["pmsm_index"].to_numpy().dtype)
    return np.concatenate(selected)


def _validate_pmsm_df(pmsm_df: pl.DataFrame) -> None:
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in pmsm_df.columns]
    if missing_columns:
        raise ValueError(f"pmsm_df is missing required columns: {missing_columns}")
    if pmsm_df.height == 0:
        return

    intensity = pl.col("median_intensity")
    invalid_intensity = pmsm_df.filter(
        intensity.is_null()
        | intensity.is_nan()
        | intensity.is_infinite()
        | (intensity < 0)
    )
    if invalid_intensity.height > 0:
        raise ValueError(
            "median_intensity must be non-null, finite, and >= 0 for every row; "
            f"found {invalid_intensity.height} invalid row(s)"
        )

    inconsistent_rt = (
        pmsm_df.group_by("precursor_index")
        .agg(pl.col("predicted_rt").n_unique().alias("_n_rt"))
        .filter(pl.col("_n_rt") > 1)
    )
    if inconsistent_rt.height > 0:
        raise ValueError(
            "each precursor_index must have exactly one predicted_rt value; "
            f"found {inconsistent_rt.height} precursor(s) with multiple values"
        )


def assign_pmsms(
    pmsm_df: pl.DataFrame,
    intensity_weight: float = 4.0,
) -> np.ndarray:
    """Select PmSMs for one run (see module docstring).

    Args:
        pmsm_df: PmSMs of a single run with `REQUIRED_COLUMNS`. A
            ``(precursor_index, cluster)`` pair may have several PmSMs. Each
            `precursor_index` needs one `predicted_rt`; `median_intensity` must
            be finite and >= 0.
        intensity_weight: Utility subtracted per 2x drop in intensity relative
            to the precursor's strongest candidate. ``0.0`` makes the DP
            raw-score only; single-precursor groups still pick by intensity.

    Returns:
        Unique selected `pmsm_index` values (input dtype).
    """
    if intensity_weight < 0:
        raise ValueError(f"intensity_weight must be >= 0, got {intensity_weight}")
    _validate_pmsm_df(pmsm_df)

    if pmsm_df.height == 0:
        return pmsm_df["pmsm_index"].to_numpy()

    utility_df = _compute_intensity_utilities(pmsm_df, intensity_weight)
    return _assign_run(utility_df)


def assign_pmsms_across_runs(
    scored_pmsm_df: pl.DataFrame,
    intensity_weight: float = 4.0,
) -> pl.DataFrame:
    """Run `assign_pmsms` per `run_index` and return the selected rows.

    `median_intensity` comes from search time (``get_pmsm_median_intensity``);
    XIC traces are not persisted for every pass, so it is not recomputed here.
    """
    # NaN median (no matched fragment at the RT center) counts as zero intensity
    scored_pmsm_df = scored_pmsm_df.with_columns(
        pl.col("median_intensity").fill_nan(0.0).fill_null(0.0)
    )

    selected_dfs = []
    for run_index, run_df in scored_pmsm_df.group_by("run_index", maintain_order=True):
        run_index = run_index[0]
        selected_pmsm_index = assign_pmsms(run_df, intensity_weight=intensity_weight)
        mask = np.isin(run_df["pmsm_index"].to_numpy(), selected_pmsm_index)
        selected_dfs.append(run_df.filter(pl.Series(mask)))

    return pl.concat(selected_dfs, how="vertical")
