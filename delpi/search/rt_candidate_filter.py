"""Cross-run observed-RT support filtering (pre-assignment PmSM pruning).

Given a table of PmSMs (Peptide-multi-Spectra-Matches) collected across
**multiple runs**, this module removes ``observed_rt`` candidates that no
other run's candidates for the same precursor support -- entirely
independent of `delpi.search.pmsm_assignment`'s per-run DP assignment (this
module never reads `score`/`cluster`/DP state, and `pmsm_assignment` never
reads anything computed here; callers run this module first, then hand the
surviving rows to `assign_pmsms` per run).

A precursor's cross-run RT distribution can be multi-modal (e.g. co-eluting
isomers, or two genuinely different elution events for the same precursor
across a batch), so this module never assumes a single representative RT, a
single Gaussian mode, or a median-RT window. Instead, every candidate's
support is a **leave-one-run-out Gaussian kernel density estimate against
every other run's candidates for the same precursor** -- a candidate is kept
if and only if runs *other than its own* collectively vote for its RT
position with enough weight, regardless of how many modes exist or how many
candidates its own run happens to contribute.

The work is split into two public steps, so callers can inspect/reuse the
raw support values without re-running the (more expensive) support
computation every time a different `min_rt_support` cutoff is tried:

- `compute_cross_run_rt_support` computes ``rt_support``/``rt_candidate_weight``
  for every row and appends them to `pmsm_df` -- it never drops any row.
- `filter_by_rt_support` drops rows using those two columns and a
  `min_rt_support` threshold, and appends an ``is_rt_supported`` column
  marking which surviving rows are trustworthy enough to feed into
  downstream quantification.
- `filter_cross_run_rt_candidates` is a thin convenience wrapper chaining
  the two, returning `pmsm_df`'s original columns plus ``is_rt_supported``.

Key design points
------------------
- RT support is computed independently per ``precursor_index`` group.
  ``precursor_index`` is already a dense id over every precursor in the
  database (targets and decoys alike, see `PeptideDatabase`), so target and
  decoy precursors never share a ``precursor_index`` -- no separate
  ``is_decoy`` grouping key is needed to keep them from supporting each
  other.
- Every run contributes the same total weight (1.0) to a precursor's RT
  density, regardless of how many candidates that run has for that
  precursor (``rt_candidate_weight = 1 / n_{p,r}``) -- otherwise a run with
  many spurious near-duplicate candidates could dominate the vote.
- A candidate's support excludes its **entire run**, not just itself, so a
  cluster of false candidates within one run can never self-support.
- A candidate that is the *only* PmSM its own run submitted for a precursor
  (``rt_candidate_weight == 1.0``) is never dropped for lacking support --
  that run made no ambiguous choice to begin with, so there's nothing for
  cross-run consistency to resolve for it.
- A candidate is also never dropped if its support isn't far below the best
  support among its *own run's* candidates for the same precursor
  (``support > min(max_support - 1, max_support * 0.5)`` within the
  ``(precursor_index, run_index)`` group, same cutoff shape as
  `delpi.search.pmsm_assignment`'s score prefilter) -- this protects a
  run's mutually-consistent, still-ambiguous candidates (e.g. near-duplicate
  peak groups) even when none of them individually clears the global
  `min_rt_support` threshold. Together with the lone-candidate exception,
  filtering only ever discards a candidate that is both one of *several*
  competing candidates from the same run *and* clearly worse than that
  run's own best candidate for this precursor.
- Groups with too few distinct runs (< ``min_precursor_runs``) are left
  completely unfiltered -- there isn't enough independent cross-run
  evidence to judge anything.
- If a group has enough runs but, for whatever reason (bad ``rt_bandwidth``/
  ``min_rt_support`` for that data), no candidate reaches the support
  threshold, the whole group is also left unfiltered rather than deleted
  outright (a safety fallback against parameter misconfiguration).
- ``is_rt_supported`` is evaluated only on the rows that survive filtering,
  and is a pure absolute check: True only if that row's own ``rt_support``
  is non-null and meets `min_rt_support` -- it doesn't care *why* a weak
  row survived (lone-candidate exception, competitive-support exception, or
  a wholly inactive/underpowered group), it always flags such rows as not
  RT-supported.
- Score, intensity, predicted RT, and any batch/condition/replicate metadata
  are never read -- this is a pure cross-run observed-RT consistency check.
"""

import numpy as np
import numba as nb
import polars as pl

REQUIRED_COLUMNS = (
    "run_index",
    "precursor_index",
    "observed_rt",
)

RT_SUPPORT_COLUMN = "rt_support"
RT_CANDIDATE_WEIGHT_COLUMN = "rt_candidate_weight"
IS_RT_SUPPORTED_COLUMN = "is_rt_supported"


def _validate_rt_bandwidth(rt_bandwidth: float) -> None:
    if not np.isfinite(rt_bandwidth) or rt_bandwidth <= 0:
        raise ValueError(f"rt_bandwidth must be finite and > 0, got {rt_bandwidth!r}")


def _validate_min_rt_support(min_rt_support: float) -> None:
    if not np.isfinite(min_rt_support) or min_rt_support <= 0:
        raise ValueError(
            f"min_rt_support must be finite and > 0, got {min_rt_support!r}"
        )


def _validate_min_precursor_runs(min_precursor_runs: int) -> None:
    if min_precursor_runs < 2:
        raise ValueError(f"min_precursor_runs must be >= 2, got {min_precursor_runs!r}")


def _validate_kernel_cutoff(kernel_cutoff: float) -> None:
    if not np.isfinite(kernel_cutoff) or kernel_cutoff <= 0:
        raise ValueError(f"kernel_cutoff must be finite and > 0, got {kernel_cutoff!r}")


def _validate_required_columns(pmsm_df: pl.DataFrame, required_columns) -> None:
    missing_columns = [c for c in required_columns if c not in pmsm_df.columns]
    if missing_columns:
        raise ValueError(f"pmsm_df is missing required columns: {missing_columns}")


def _validate_observed_rt_numeric(pmsm_df: pl.DataFrame) -> None:
    if not pmsm_df.schema["observed_rt"].is_numeric():
        raise ValueError(
            "observed_rt must be a numeric column, got dtype "
            f"{pmsm_df.schema['observed_rt']}"
        )


def _validate_no_nulls(pmsm_df: pl.DataFrame, columns) -> None:
    for col in columns:
        if pmsm_df[col].null_count() > 0:
            raise ValueError(f"{col} must not contain null values")


@nb.njit(cache=True, nogil=True)
def _compute_leave_one_run_out_support(
    group_start: np.ndarray,
    group_end: np.ndarray,
    run_index: np.ndarray,
    observed_rt: np.ndarray,
    run_candidate_weight: np.ndarray,
    rt_bandwidth: np.float32,
    kernel_cutoff: np.float32,
) -> np.ndarray:
    """Leave-one-run-out Gaussian-kernel RT support for every candidate.

    Inputs are flat arrays covering every eligible precursor group
    concatenated together, RT-sorted *within* each group; `group_start`/
    `group_end` (both length ``n_groups``) delimit each group's ``[start,
    end)`` slice. Candidate ``i``'s support sums
    ``run_candidate_weight[j] * exp(-((t_i - t_j) / rt_bandwidth)^2 / 2)``
    over every other candidate ``j`` of the *same* group whose ``run_index``
    differs from candidate ``i``'s (the whole run is excluded, not just row
    ``i``); since each group's rows are already RT-sorted, the scan in both
    directions from ``i`` stops as soon as ``|t_i - t_j| >
    kernel_cutoff * rt_bandwidth``, so unrelated far-away candidates are
    never visited.
    """
    n = observed_rt.shape[0]
    support = np.zeros(n, dtype=np.float32)
    cutoff_distance = kernel_cutoff * rt_bandwidth

    for g in range(group_start.shape[0]):
        s = group_start[g]
        e = group_end[g]
        for i in range(s, e):
            t_i = observed_rt[i]
            r_i = run_index[i]
            total = np.float32(0.0)

            j = i - 1
            while j >= s:
                dt = t_i - observed_rt[j]
                if dt > cutoff_distance:
                    break
                if run_index[j] != r_i:
                    u = dt / rt_bandwidth
                    total += run_candidate_weight[j] * np.exp(-0.5 * u * u)
                j -= 1

            j = i + 1
            while j < e:
                dt = observed_rt[j] - t_i
                if dt > cutoff_distance:
                    break
                if run_index[j] != r_i:
                    u = dt / rt_bandwidth
                    total += run_candidate_weight[j] * np.exp(-0.5 * u * u)
                j += 1

            support[i] = total

    return support


def _empty_support_columns() -> list:
    return [
        pl.lit(None, dtype=pl.Float32).alias(RT_SUPPORT_COLUMN),
        pl.lit(None, dtype=pl.Float32).alias(RT_CANDIDATE_WEIGHT_COLUMN),
    ]


def compute_cross_run_rt_support(
    pmsm_df: pl.DataFrame,
    *,
    rt_bandwidth: float,
    min_precursor_runs: int = 3,
    kernel_cutoff: float = 4.0,
) -> pl.DataFrame:
    """Append leave-one-run-out RT support columns to every row of `pmsm_df`.

    Parameters
    ----------
    pmsm_df:
        PmSMs from any number of runs; must contain `REQUIRED_COLUMNS`
        (extra columns are preserved untouched). Not mutated.
    rt_bandwidth:
        Gaussian kernel bandwidth ``h``, in the same units as
        ``observed_rt``. Dataset-dependent -- callers must pass it
        explicitly.
    min_precursor_runs:
        A ``precursor_index`` group's candidates only get a real
        `RT_SUPPORT_COLUMN` value if at least this many distinct runs
        contribute to that group (default 3; must be >= 2, since a single
        run can never leave-one-run-out support anything).
    kernel_cutoff:
        Candidates farther than ``kernel_cutoff * rt_bandwidth`` apart
        contribute exactly 0 kernel weight to each other (default 4.0).

    Returns
    -------
    pl.DataFrame
        `pmsm_df`'s original columns/dtypes/row order, plus 2 appended
        ``Float32`` columns: `RT_SUPPORT_COLUMN` (the leave-one-run-out
        support ``S_i``, null wherever its precursor group has fewer than
        `min_precursor_runs` distinct runs) and `RT_CANDIDATE_WEIGHT_COLUMN`
        (``1 / n_{p,r}``, always populated, needed by `filter_by_rt_support`'s
        lone-candidate exception). No row is dropped.
    """
    _validate_rt_bandwidth(rt_bandwidth)
    _validate_min_precursor_runs(min_precursor_runs)
    _validate_kernel_cutoff(kernel_cutoff)
    _validate_required_columns(pmsm_df, REQUIRED_COLUMNS)
    _validate_observed_rt_numeric(pmsm_df)
    _validate_no_nulls(pmsm_df, ("run_index", "precursor_index"))

    if pmsm_df.height == 0:
        return pmsm_df.with_columns(*_empty_support_columns())

    original_columns = pmsm_df.columns

    df = pmsm_df.with_row_index("_orig_order").with_columns(
        (1.0 / pl.len().over(["precursor_index", "run_index"]))
        .cast(pl.Float32)
        .alias(RT_CANDIDATE_WEIGHT_COLUMN),
        pl.col("run_index").n_unique().over("precursor_index").alias("_n_runs"),
    )

    eligible_df = df.filter(pl.col("_n_runs") >= min_precursor_runs).sort(
        ["precursor_index", "observed_rt"]
    )

    if eligible_df.height == 0:
        # no precursor group anywhere has enough cross-run evidence
        return pmsm_df.with_columns(*_empty_support_columns())

    eligible_df = eligible_df.with_row_index("_pos")
    bounds_df = eligible_df.group_by("precursor_index", maintain_order=True).agg(
        pl.col("_pos").min().alias("_start"),
        (pl.col("_pos").max() + 1).alias("_end"),
    )

    support_arr = _compute_leave_one_run_out_support(
        bounds_df["_start"].to_numpy(),
        bounds_df["_end"].to_numpy(),
        eligible_df["run_index"].to_numpy(),
        eligible_df["observed_rt"].to_numpy(),
        eligible_df[RT_CANDIDATE_WEIGHT_COLUMN].to_numpy(),
        np.float32(rt_bandwidth),
        np.float32(kernel_cutoff),
    )

    support_df = eligible_df.with_columns(
        pl.Series(name=RT_SUPPORT_COLUMN, values=support_arr, dtype=pl.Float32)
    ).select(["_orig_order", RT_SUPPORT_COLUMN])

    return (
        df.join(support_df, on="_orig_order", how="left")
        .sort("_orig_order")
        .select([*original_columns, RT_SUPPORT_COLUMN, RT_CANDIDATE_WEIGHT_COLUMN])
    )


def filter_by_rt_support(
    pmsm_df: pl.DataFrame,
    *,
    min_rt_support: float,
    support_column: str = RT_SUPPORT_COLUMN,
    weight_column: str = RT_CANDIDATE_WEIGHT_COLUMN,
    is_rt_supported_column: str = IS_RT_SUPPORTED_COLUMN,
) -> pl.DataFrame:
    """Drop PmSM candidates using RT support columns from
    `compute_cross_run_rt_support` (or equivalently-shaped columns), and
    flag which survivors are trustworthy enough for quantification.

    Parameters
    ----------
    pmsm_df:
        Must contain ``precursor_index``, ``run_index``, `support_column`
        and `weight_column` (extra columns are preserved untouched). Not
        mutated.
    min_rt_support:
        Minimum `support_column` value required to keep a candidate.
        Dataset-dependent -- callers must pass it explicitly.
    support_column, weight_column:
        Column names produced by `compute_cross_run_rt_support`.
    is_rt_supported_column:
        Name of the appended Boolean column (see Returns).

    Returns
    -------
    pl.DataFrame
        `pmsm_df`'s surviving rows (column order/dtypes/relative row order
        preserved) plus one appended Boolean `is_rt_supported_column`. A
        precursor group is only filtered if it has >= 1 candidate reaching
        `min_rt_support` (otherwise every row of that group -- including
        any with a null `support_column` -- is kept as a safety fallback);
        within a filtered group, a candidate is dropped only if its support
        is below `min_rt_support`, it isn't the lone candidate its run
        submitted for that precursor (`weight_column` == 1.0), *and* it
        isn't close to the best support among its own run's candidates for
        this precursor (``support > min(max_support - 1, max_support * 0.5)``
        within the same ``(precursor_index, run_index)`` group).
        `is_rt_supported_column` is computed on the surviving rows only, and
        is True only where `support_column` is non-null and meets
        `min_rt_support` -- False for every row kept solely via the
        lone-candidate exception, the competitive-support exception, or an
        inactive/underpowered group. An empty `pmsm_df` returns an empty
        frame with the same schema plus `is_rt_supported_column`.
    """
    _validate_min_rt_support(min_rt_support)
    _validate_required_columns(
        pmsm_df, ("precursor_index", "run_index", support_column, weight_column)
    )
    _validate_no_nulls(pmsm_df, ("precursor_index",))

    if pmsm_df.height == 0:
        return pmsm_df.with_columns(
            pl.lit(False, dtype=pl.Boolean).alias(is_rt_supported_column)
        ).clear()

    meets_threshold = (pl.col(support_column) >= min_rt_support).fill_null(False)
    # a run that only ever submitted 1 PmSM for this precursor already made
    # its only choice -- never penalize it for lacking outside support, only
    # within-run ambiguity (>1 candidate from the same run) is resolved here
    is_lone_candidate = (pl.col(weight_column) == 1.0).fill_null(False)
    group_active = meets_threshold.any().over("precursor_index")

    # a candidate is also never dropped if it's close to its own run's best
    # candidate for this precursor -- protects a run's mutually-consistent,
    # still-ambiguous candidates even when none of them individually clears
    # min_rt_support; the left OR-operand is null (not False) whenever
    # max_support itself is null (group too small to compute support at all),
    # which is harmless since group_active is already False for that group
    max_support = pl.col(support_column).max().over(["precursor_index", "run_index"])
    support_gap = max_support - pl.col(support_column)
    competitive_support = (support_gap < 1.0) | (
        pl.col(support_column) > max_support * 0.5
    ).fill_null(False)

    drop_expr = (
        group_active
        & (~meets_threshold)
        & (~is_lone_candidate)
        & (~competitive_support)
    )
    # evaluated on survivors only -- a pure absolute check, independent of
    # *why* a weak row survived (lone-candidate or competitive-support exception)
    is_rt_supported_expr = meets_threshold & pl.col(support_column).is_not_null()

    return pmsm_df.filter(~drop_expr).with_columns(
        is_rt_supported_expr.alias(is_rt_supported_column)
    )


def filter_cross_run_rt_candidates(
    pmsm_df: pl.DataFrame,
    *,
    rt_bandwidth: float,
    min_rt_support: float,
    min_precursor_runs: int = 3,
    kernel_cutoff: float = 4.0,
) -> pl.DataFrame:
    """Convenience wrapper chaining `compute_cross_run_rt_support` and
    `filter_by_rt_support`; see both for parameter/behavior details.

    Returns
    -------
    pl.DataFrame
        `pmsm_df`'s surviving rows, with the original column order, dtypes
        and relative row order preserved, plus an appended
        `IS_RT_SUPPORTED_COLUMN`. An empty `pmsm_df` returns an empty frame
        with the same schema plus `IS_RT_SUPPORTED_COLUMN`.
    """
    if pmsm_df["run_index"].n_unique() <= min_precursor_runs:
        return pmsm_df.with_columns(pl.lit(True).alias(IS_RT_SUPPORTED_COLUMN))

    original_columns = pmsm_df.columns
    annotated_df = compute_cross_run_rt_support(
        pmsm_df,
        rt_bandwidth=rt_bandwidth,
        min_precursor_runs=min_precursor_runs,
        kernel_cutoff=kernel_cutoff,
    )
    filtered_df = filter_by_rt_support(annotated_df, min_rt_support=min_rt_support)
    return filtered_df.select([*original_columns, IS_RT_SUPPORTED_COLUMN])
