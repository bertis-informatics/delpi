import polars as pl
import numpy as np

from delpi.search.dia.lfq_utils import _nb_build_L_b


def _maxlfq_one_protein(
    df_p: pl.DataFrame,
    run_col: str,
    peptide_col: str,
    log_col: str = "logI",
) -> tuple[list, np.ndarray]:
    """
    Cox 2014 MaxLFQ

    Parameters
    ----------
    df_p : pl.DataFrame (columns: run_col, peptide_col, log_col)
    run_col : str
    peptide_col : str
    log_col : str
        log-intensity column name (e.g. "logI")

    Returns
    -------
    runs : run identifier list (in fixed order)
    x : np.ndarray
    """
    # List of runs observed in this protein (maintain order)
    runs = df_p[run_col].unique().to_list()
    n_runs = len(runs)
    if n_runs == 0:
        return [], np.array([], dtype=float)
    if n_runs == 1:
        # If only one run, cannot compute ratios, so use mean log-intensity of that run
        mean_log = df_p[log_col].mean()
        return runs, np.array([mean_log], dtype=float)

    # Mapping: run -> 0..n_runs-1
    run_to_idx = {r: i for i, r in enumerate(runs)}
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

    L, b = _nb_build_L_b(n_runs, pep_idx_sorted, run_idx_sorted, logI_sorted)

    # gauge fixing: x[0] = 0
    L[0, :] = 0.0
    L[:, 0] = 0.0
    L[0, 0] = 1.0
    b[0] = 0.0

    # Solve linear system (use solve if possible, otherwise lstsq)
    try:
        x = np.linalg.solve(L, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(L, b, rcond=None)

    return runs, x


def maxlfq(
    df: pl.DataFrame,
    protein_col: str = "protein_group",
    peptide_col: str = "peptide_index",
    run_col: str = "run_index",
    intensity_col: str = "peptide_abundance",  # ms2_area-like value
    min_peptides_per_protein: int = 2,
) -> pl.DataFrame:
    """
    Implementation of Cox et al. 2014 MaxLFQ algorithm (pairwise log-ratio + least squares)
    based on polars DataFrame.

    - Performs MaxLFQ per protein to estimate run-wise protein log-abundance
    - Scales based on the global median of log(intensity) from the entire dataset,
      finally returning protein intensities in a range similar to the original intensity (ms2_area).

    Parameters
    ----------
    df : pl.DataFrame
        Long-format table with protein, peptide, run, intensity.
    protein_col, peptide_col, run_col, intensity_col : str
        Column names.
    min_peptides_per_protein : int
        Require at least this many distinct peptides per protein
        (small proteins with too few peptides are unsuitable for MaxLFQ and should be excluded or set to NaN).

    Returns
    -------
    pl.DataFrame
        Wide-format protein intensity matrix:
        rows = protein, columns = runs (MaxLFQ-based protein_abundance).
    """
    # 0) Filter out intensity <= 0 and NaN
    df = df.filter(pl.col(intensity_col).is_not_null() & (pl.col(intensity_col) > 0))

    protein_dtype = df.schema[protein_col]
    result_schema = {
        protein_col: protein_dtype,
        run_col: pl.UInt32,
        "abundance": pl.Float32,
    }

    if df.height == 0:
        return pl.DataFrame(schema=result_schema)

    # 1) Add log-intensity
    df_log = df.with_columns(pl.col(intensity_col).log().alias("logI"))

    # Global log-intensity median from entire dataset (for absolute scale reference)
    global_log_median = float(df_log.select(pl.col("logI").median()).item())

    # 2) Perform MaxLFQ per protein
    records: list[tuple] = []

    for prot, df_p in df_log.group_by(protein_col, maintain_order=False):
        # Filter by peptide count per protein
        n_pep = df_p.select(pl.col(peptide_col).n_unique()).item()
        if n_pep < min_peptides_per_protein:
            # If too few peptides, stable estimation via MaxLFQ is difficult.
            # Could skip or use simple sum/mean as an option (can be added later).
            continue

        runs, x = _maxlfq_one_protein(
            df_p=df_p,
            run_col=run_col,
            peptide_col=peptide_col,
            log_col="logI",
        )

        if len(runs) == 0:
            continue

        # x is run-wise log-abundance (relative, gauge x[0]=0).
        x = np.asarray(x, dtype=float)

        if len(runs) == 1:
            # For single run, x already contains the mean log-intensity of peptides
            # No need to center (would result in 0), just use it directly
            log_protein = x
        else:
            # For multiple runs, center so that protein-level mean is 0,
            # then add global_log_median to match scale
            x_centered = x - np.nanmean(x)
            log_protein = x_centered + global_log_median

        protein_intensity = np.exp(log_protein)

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
