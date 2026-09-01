import polars as pl
import numpy as np

from delpi.search.dia.lfq_utils import _nb_build_L_b, _nb_maxlfq_all_proteins


def _connected_components_from_L(L: np.ndarray, n_runs: int) -> list[list[int]]:
    """Connected components of the run graph encoded by L's off-diagonal pattern.

    Off-diagonal entries of L are non-zero exactly for run pairs that had a
    sufficiently supported (>= min_ratio_count) MaxLFQ ratio, so this reflects
    which runs are linked by at least one precursor-ratio path.
    """
    visited = np.zeros(n_runs, dtype=bool)
    components: list[list[int]] = []
    for start in range(n_runs):
        if visited[start]:
            continue
        visited[start] = True
        comp = [start]
        stack = [start]
        while stack:
            node = stack.pop()
            for nb_idx in np.nonzero(L[node])[0]:
                if nb_idx != node and not visited[nb_idx]:
                    visited[nb_idx] = True
                    comp.append(int(nb_idx))
                    stack.append(int(nb_idx))
        components.append(comp)
    return components


def _maxlfq_one_protein(
    df_p: pl.DataFrame,
    run_col: str,
    peptide_col: str,
    intensity_col: str,
    log_col: str = "logI",
    min_ratio_count: int = 2,
) -> tuple[list, np.ndarray]:
    """
    Cox 2014 MaxLFQ.

    The run graph (edges = run pairs with >= min_ratio_count shared-precursor
    ratios) is split into connected components. Each component is solved and
    gauge-fixed independently, then rescaled to a linear-scale profile using
    only the observed precursor intensities of the runs in that component.
    No ratio is inferred between disconnected components.

    Parameters
    ----------
    df_p : pl.DataFrame (columns: run_col, peptide_col, intensity_col, log_col)
    run_col : str
    peptide_col : str
    intensity_col : str
        linear-scale intensity column name (used for rescaling)
    log_col : str
        log-intensity column name (e.g. "logI")
    min_ratio_count : int
        minimum number of shared-precursor ratio observations required for a
        run-pair median ratio to be used in the least-squares system

    Returns
    -------
    runs : run identifier list (in fixed order)
    protein_intensity : np.ndarray
        linear-scale abundance aligned with runs
    """
    # List of runs observed in this protein (maintain order)
    runs = df_p[run_col].unique().to_list()
    n_runs = len(runs)
    if n_runs == 0:
        return [], np.array([], dtype=float)

    run_to_idx = {r: i for i, r in enumerate(runs)}

    # Per-run observed intensity totals, used for rescaling
    intensity_by_run = np.zeros(n_runs, dtype=float)
    for r, total in df_p.group_by(run_col).agg(pl.col(intensity_col).sum()).iter_rows():
        intensity_by_run[run_to_idx[r]] = total

    if n_runs == 1:
        # Single run: no ratio can be computed, use the summed observed intensity
        return runs, intensity_by_run

    # Mapping: peptide -> 0..n_pep-1
    pep_vals = df_p[peptide_col].to_list()
    pep_to_idx = {}
    pep_idx_list = []
    next_idx = 0
    for v in pep_vals:
        idx = pep_to_idx.get(v)
        if idx is None:
            pep_to_idx[v] = next_idx
            pep_idx_list.append(next_idx)
            next_idx += 1
        else:
            pep_idx_list.append(idx)

    run_idx_arr = np.array(
        [run_to_idx[v] for v in df_p[run_col].to_list()], dtype=np.int64
    )
    pep_idx_arr = np.array(pep_idx_list, dtype=np.int64)
    logI_arr = np.array(df_p[log_col].to_list(), dtype=np.float64)

    # Sort by peptide index for consecutive grouping (for numba group boundary scan)
    order = np.argsort(pep_idx_arr, kind="mergesort")  # stable sort
    pep_idx_sorted = pep_idx_arr[order]
    run_idx_sorted = run_idx_arr[order]
    logI_sorted = logI_arr[order]

    L, b = _nb_build_L_b(
        n_runs, pep_idx_sorted, run_idx_sorted, logI_sorted, min_ratio_count
    )

    protein_intensity = np.zeros(n_runs, dtype=float)

    for comp in _connected_components_from_L(L, n_runs):
        if len(comp) == 1:
            # No ratio connects this run to any other: use its own observed intensity
            protein_intensity[comp[0]] = intensity_by_run[comp[0]]
            continue

        idx = np.array(comp, dtype=np.int64)
        L_sub = L[np.ix_(idx, idx)].copy()
        b_sub = b[idx].copy()

        # gauge fixing within this component: x_sub[0] = 0
        L_sub[0, :] = 0.0
        L_sub[:, 0] = 0.0
        L_sub[0, 0] = 1.0
        b_sub[0] = 0.0

        # Solve linear system (use solve if possible, otherwise lstsq)
        try:
            x_sub = np.linalg.solve(L_sub, b_sub)
        except np.linalg.LinAlgError:
            x_sub, *_ = np.linalg.lstsq(L_sub, b_sub, rcond=None)

        # Convert to a relative linear-scale profile (numerically stable), then
        # rescale by this component's own observed intensity total so that
        # sum(protein_intensity) == sum(observed intensity) within the component.
        relative = np.exp(x_sub - np.max(x_sub))
        total_intensity = intensity_by_run[idx].sum()
        protein_intensity[idx] = relative * total_intensity / relative.sum()

    return runs, protein_intensity


def maxlfq_legacy(
    df: pl.DataFrame,
    protein_col: str = "protein_group",
    peptide_col: str = "peptide_index",
    run_col: str = "run_index",
    intensity_col: str = "peptide_abundance",  # ms2_area-like value
    min_peptides_per_protein: int = 2,
    min_ratio_count: int = 2,
) -> pl.DataFrame:
    """
    Implementation of Cox et al. 2014 MaxLFQ algorithm (pairwise log-ratio + least squares)
    based on polars DataFrame.

    - Performs MaxLFQ per protein to estimate the relative run-wise protein
      abundance profile, then rescales it (single multiplicative factor per
      protein, or per connected component - see below) so that the sum
      across runs matches the observed total peptide intensity, preserving
      protein-to-protein abundance differences.
    - A protein's run graph may be disconnected (no shared-precursor ratio
      path between some runs). Each connected component is solved,
      gauge-fixed, and rescaled independently, using only the observed
      intensities of the runs within that component; no ratio is inferred
      between disconnected components.

    Parameters
    ----------
    df : pl.DataFrame
        Long-format table with protein, peptide, run, intensity.
    protein_col, peptide_col, run_col, intensity_col : str
        Column names.
    min_peptides_per_protein : int
        Require at least this many distinct peptides per protein
        (small proteins with too few peptides are unsuitable for MaxLFQ and should be excluded or set to NaN).
    min_ratio_count : int
        Minimum number of shared-precursor ratio observations required for a
        run-pair median ratio to be used in the least-squares system.

    Returns
    -------
    pl.DataFrame
        Wide-format protein intensity matrix:
        rows = protein, columns = runs (MaxLFQ-based protein_abundance).
    """
    # 0) Filter out intensity <= 0 and NaN
    df = df.filter(pl.col(intensity_col).is_not_null() & (pl.col(intensity_col) > 0))

    protein_dtype = df.schema[protein_col]
    run_dtype = df.schema[run_col]
    result_schema = {
        protein_col: protein_dtype,
        run_col: run_dtype,
        "maxlfq_abundance": pl.Float32,
    }

    if df.height == 0:
        return pl.DataFrame(schema=result_schema)

    # 1) Add log-intensity
    df_log = df.with_columns(pl.col(intensity_col).log().alias("logI"))

    # 2) Perform MaxLFQ per protein
    records: list[tuple] = []

    for prot, df_p in df_log.group_by(protein_col, maintain_order=False):
        # Filter by peptide count per protein
        n_pep = df_p.select(pl.col(peptide_col).n_unique()).item()
        if n_pep < min_peptides_per_protein:
            # If too few peptides, stable estimation via MaxLFQ is difficult.
            # Could skip or use simple sum/mean as an option (can be added later).
            continue

        runs, protein_intensity = _maxlfq_one_protein(
            df_p=df_p,
            run_col=run_col,
            peptide_col=peptide_col,
            intensity_col=intensity_col,
            log_col="logI",
            min_ratio_count=min_ratio_count,
        )

        if len(runs) == 0:
            continue

        for run, val in zip(runs, protein_intensity):
            records.append((prot[0], run, float(val)))

    if not records:
        return pl.DataFrame(schema=result_schema)

    df_prot_long = pl.DataFrame(records, schema=result_schema, orient="row")

    return df_prot_long

    # 3) Pivot to protein × run wide matrix
    # protein_matrix = (
    #     df_prot_long
    #     .pivot(
    #         values="protein_abundance",
    #         index=protein_col,
    #         on=run_col,
    #     )
    #     .sort(protein_col)
    # )

    # return protein_matrix


def maxlfq(
    df: pl.DataFrame,
    protein_col: str = "protein_group",
    peptide_col: str = "peptide_index",
    run_col: str = "run_index",
    intensity_col: str = "peptide_abundance",
    min_peptides_per_protein: int = 2,
    min_ratio_count: int = 2,
) -> pl.DataFrame:
    """
    Vectorized MaxLFQ, functionally equivalent to `maxlfq()`.

    `maxlfq()` re-enters Polars/Python once per protein (group_by iteration
    plus several per-protein `.to_list()`/`group_by().agg()` calls and Python
    dict-based factorization), which dominates runtime as the number of
    proteins/runs grows. Here, the entire per-protein loop - run/peptide
    grouping, ratio construction, connected components, per-component solve
    and rescale - runs inside a single compiled `_nb_maxlfq_all_proteins`
    kernel, with only one Polars pass to prepare inputs and one to assemble
    the output.

    See `maxlfq()` for parameter and return value semantics.
    """
    df = df.filter(pl.col(intensity_col).is_not_null() & (pl.col(intensity_col) > 0))

    protein_dtype = df.schema[protein_col]
    run_dtype = df.schema[run_col]
    result_schema = {
        protein_col: protein_dtype,
        run_col: run_dtype,
        "maxlfq_abundance": pl.Float32,
    }

    if df.height == 0:
        return pl.DataFrame(schema=result_schema)

    # Dense int32 codes (any orderable dtype works, protein/run/peptide counts
    # always fit int32) sorted so each protein's rows are contiguous and
    # peptide-ordered, as required by _nb_build_L_b. logI/intensity are kept
    # float32 since they're the other arrays sized by the full row count.
    df = df.with_columns(
        (pl.col(protein_col).rank(method="dense").cast(pl.Int32) - 1).alias(
            "__protein_idx"
        ),
        (pl.col(run_col).rank(method="dense").cast(pl.Int32) - 1).alias("__run_idx"),
        (pl.col(peptide_col).rank(method="dense").cast(pl.Int32) - 1).alias(
            "__pep_idx"
        ),
        pl.col(intensity_col).log().cast(pl.Float32).alias("__logI"),
        pl.col(intensity_col).cast(pl.Float32).alias("__intensity"),
    ).sort(["__protein_idx", "__pep_idx"])

    # code -> original value lookup tables
    protein_lookup = (
        df.select(protein_col, "__protein_idx")
        .unique()
        .sort("__protein_idx")[protein_col]
        .to_numpy()
    )
    run_lookup = (
        df.select(run_col, "__run_idx").unique().sort("__run_idx")[run_col].to_numpy()
    )
    n_runs_total = int(df["__run_idx"].max()) + 1

    out_protein_idx, out_run_idx, out_abundance = _nb_maxlfq_all_proteins(
        df["__protein_idx"].to_numpy(),
        df["__pep_idx"].to_numpy(),
        df["__run_idx"].to_numpy(),
        df["__logI"].to_numpy(),
        df["__intensity"].to_numpy(),
        n_runs_total,
        min_peptides_per_protein,
        min_ratio_count,
    )

    if out_protein_idx.shape[0] == 0:
        return pl.DataFrame(schema=result_schema)

    return pl.DataFrame(
        {
            protein_col: protein_lookup[out_protein_idx],
            run_col: run_lookup[out_run_idx],
            "maxlfq_abundance": out_abundance,
        },
        schema=result_schema,
    )
