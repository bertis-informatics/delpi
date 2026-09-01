"""Joint, precursor-relative-intensity-aware PmSM assignment.

Given a table of PmSMs (Peptide-multi-Spectra-Matches, the common match unit
for both DDA and DIA search results in DelPi) for a **single run**, this
module selects a final PmSM subset by solving a single joint,
order-preserving dynamic program that simultaneously decides:

- which precursors to use,
- which PmSM ``cluster`` (a run-local fragment-peak-sharing competition group,
  already defined upstream) to use,
- the one-to-one precursor <-> cluster assignment, and
- the final selected PmSM for each assigned pair.

No precursor or cluster is preselected before the DP: every precursor and
every cluster in an ``alignment_group`` (a connected component of the
precursor_index <-> cluster bipartite graph, computed fresh in this module)
is a DP candidate.

When a precursor has several candidate PmSMs, `score` alone can favor a
candidate whose ``median_intensity`` is far weaker than another candidate of
the same precursor. To avoid that:

- alignment groups with a **single** candidate precursor select the PmSM
  with the highest `median_intensity` directly -- `score` is not used at all,
  since there is no competing precursor to weigh it against.
- alignment groups with **two or more** precursors run the joint DP over an
  assignment utility that combines `score` with the PmSM's *precursor-relative*
  log2 intensity (relative to the strongest candidate of the same
  `precursor_index`). Because the normalization denominator is always the
  precursor's own maximum intensity, absolute abundance differences between
  *different* precursors never act as an assignment bonus.

Callers processing multiple runs must invoke `assign_pmsms` once per run
(this module has no cross-run, replicate, condition, or quantity awareness).
Target and decoy PmSMs are treated identically (this runs before FDR
control) -- ``is_decoy`` is never read (and need not be present).
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

# DP state ties (total_assignment_utility, total_raw_score) are compared with this tolerance
_TIE_TOLERANCE = 1e-8
# stabilizes the precursor-relative log2 intensity ratio when intensities are 0
_INTENSITY_EPSILON = 1e-6


# --------------------------------------------------------------------------- #
# union-find (plain, no external graph library)
# --------------------------------------------------------------------------- #


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
    """Union-find over precursor nodes ``[0, n_precursor_nodes)`` and cluster
    nodes ``[n_precursor_nodes, n_precursor_nodes + n_cluster_nodes)``, unioning
    the two endpoints of every PmSM row. Returns, for every node, the (non-dense)
    root id of its connected component.
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
    """Sorted unique values and inverse indices, i.e. a numba-compatible
    equivalent of ``np.unique(values, return_inverse=True)`` (numba's
    ``np.unique`` does not support ``return_inverse``).
    """
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


# --------------------------------------------------------------------------- #
# joint order-preserving assignment DP
# --------------------------------------------------------------------------- #


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
    """Whether DP state A beats state B under (match count, then utility, then
    raw score), each of utility/score compared with tolerance `tol` (count is
    exact and compared first, so a valid match is never dropped merely
    because its assignment utility is negative). Returns False on an exact
    tie, so the caller's evaluation order decides the winner.
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
    """Joint order-preserving assignment DP for one alignment group.

    `utility`/`raw_score`/`valid` are ``(M, N)`` matrices over precursors
    (rows, ``predicted_rt`` order) and clusters (cols, representative
    ``frame_num`` order); `valid[i, j]` marks that a PmSM connects precursor
    `i` and cluster `j` (skipping a precursor or a cluster always has utility
    and raw score 0). Maximizes the number of matches first, then total
    `utility`, then total `raw_score`; remaining exact ties are broken
    deterministically by preferring, in order, a match over a precursor skip
    over a cluster skip.

    Returns a ``(k, 2)`` array of 0-based ``(precursor_row, cluster_col)`` matches.
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
    """Run `_dp_assign` once per alignment group over flat, group-sorted arrays.

    `group_start`/`group_end` delimit each group's rows in the (already
    sorted-by-group) `precursor_order`/`cluster_order`/`assignment_utility`/
    `raw_score`/`pmsm_index` arrays. Returns the selected `pmsm_index` values
    (all groups concatenated).
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
        # unfilled cells are never read (only `valid` positions get selected), so 0 is a safe filler
        pmsm_grid = np.zeros((m, n), dtype=pmsm_index.dtype)
        for k in range(s, e):
            i, j = p_local[k - s], c_local[k - s]
            utility[i, j] = assignment_utility[k]
            raw[i, j] = raw_score[k]
            valid[i, j] = True
            pmsm_grid[i, j] = pmsm_index[k]

        for i, j in _dp_assign(utility, raw, valid, tol):
            out[out_cnt] = pmsm_grid[i, j]
            out_cnt += 1
    return out[:out_cnt]


# --------------------------------------------------------------------------- #
# Polars-level pipeline
# --------------------------------------------------------------------------- #


def _compute_intensity_utilities(
    pmsm_df: pl.DataFrame, intensity_weight: float
) -> pl.DataFrame:
    """Add ``_relative_log_intensity`` and ``_assignment_utility`` columns to
    every PmSM row; ``score`` and ``median_intensity`` themselves are never
    modified.

    ``_relative_log_intensity`` expresses a PmSM's ``median_intensity`` as a
    log2 ratio to the strongest candidate of the same
    ``(run_index, precursor_index)`` -- always <= 0, and exactly 0 for that
    strongest candidate. Because the denominator is always the precursor's
    own maximum, absolute intensity differences between *different*
    precursors never enter the utility (precursor-relative normalization,
    not raw ``log2(median_intensity)``).
    """
    max_intensity = (
        pl.col("median_intensity").max().over("precursor_index")
    )
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
    """Return `run_df` with an added ``_alignment_group`` column: the
    (dense, run-local) connected component id of the
    ``precursor_index <-> cluster`` bipartite graph built from every PmSM row
    in this run (plain union-find; `cluster` and `_alignment_group` are
    distinct concepts and never mixed).
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
        node_df["_precursor_node"].to_numpy(),
        node_df["_cluster_node"].to_numpy(),
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
    """Build the per-alignment-group DP input: one row per PmSM candidate
    edge, with `alignment_group`, `precursor_order` (``predicted_rt`` asc,
    then `precursor_index` asc) and `cluster_order` (median `frame_num` asc,
    then `cluster` asc) attached. Every precursor and cluster in the run is a
    candidate; there is no preselection step.
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
    """Solve the joint order-preserving assignment DP independently per
    alignment group. `alignment_group` must be sorted ascending so each
    group's rows are contiguous (as produced by `_prepare_assignment_problem`).

    Returns the selected `pmsm_index` values (all groups concatenated).
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
    """For alignment groups with exactly one candidate precursor, there is no
    competing precursor to weigh against, so `score` is not used at all: the
    winning PmSM is simply the one with the highest `median_intensity` (ties
    broken by `score` desc, then `pmsm_index` asc, for determinism).
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
    """Run the full per-run pipeline (alignment groups -> ordering -> PmSM
    selection) and return the selected `pmsm_index` values for this run.

    Alignment groups with a single candidate precursor are selected directly
    by `median_intensity` (see `_select_single_precursor_groups`); groups with
    two or more precursors are solved with the full order-preserving joint DP
    over `_assignment_utility` (`score + intensity_weight * relative_log_intensity`).
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

    duplicate_edges = (
        pmsm_df.group_by(["precursor_index", "cluster"])
        .agg(pl.len().alias("_n"))
        .filter(pl.col("_n") > 1)
    )
    if duplicate_edges.height > 0:
        raise ValueError(
            "pmsm_df must have at most one PmSM per (precursor_index, cluster); "
            f"found {duplicate_edges.height} duplicated combination(s)"
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
    """Select, for **one run**, which precursors and PmSM clusters to use and
    the one-to-one PmSM assignment between them, via a single order-preserving
    dynamic program (see module docstring).

    Args:
        pmsm_df: PmSM table for a single run, with at least the columns in
            `REQUIRED_COLUMNS`. Each ``(precursor_index, cluster)``
            combination must have at most one PmSM, each `precursor_index`
            must have exactly one `predicted_rt` value, and `median_intensity`
            must be non-null, finite, and >= 0 for every row. Any
            ``run_index``, ``observed_rt`` or ``is_decoy`` column, if
            present, is ignored.
        intensity_weight: Amount subtracted from `score` per 2x decrease in a
            candidate's intensity relative to the strongest candidate of the
            same precursor. ``0.0`` disables intensity entirely (raw-score-
            only joint assignment). Must be ``>= 0``.

    Returns:
        The selected `pmsm_index` values for this run as a 1D NumPy array
        (same dtype as the input `pmsm_index` column). Every value is unique
        and present in `pmsm_df`. Empty input yields an empty array.
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
    """Convenience wrapper that runs `assign_pmsms` for every run in
    `scored_pmsm_df` (grouped by `run_index`).

    `median_intensity` is expected to already be a column of
    `scored_pmsm_df` (computed once, per PmSM, at search time -- see
    ``get_pmsm_median_intensity`` calls in the DIA/DDA search engines --
    rather than recomputed here from XIC traces, which may not even be
    persisted for every search pass). This is the only place in the
    assignment pipeline that is aware of multiple runs; `assign_pmsms`
    itself always processes a single run in isolation.

    Args:
        scored_pmsm_df: Scored PmSMs for one or more runs, with at least
            `REQUIRED_COLUMNS` plus `run_index`.
        intensity_weight: Forwarded to `assign_pmsms` (see its docstring).

    Returns:
        The subset of `scored_pmsm_df` rows selected by the per-run
        assignment (all original columns preserved).
    """
    # a PmSM with no matched fragment at the RT center has no defined median
    # (get_pmsm_median_intensity returns NaN for it); treat it as zero
    # intensity rather than rejecting it in `_validate_pmsm_df`.
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
