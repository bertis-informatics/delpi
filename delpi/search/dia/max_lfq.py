import polars as pl
import numpy as np

from delpi.search.dia.lfq_utils import _nb_maxlfq_all_proteins


def maxlfq(
    df: pl.DataFrame,
    protein_col: str = "protein_group",
    peptide_col: str = "precursor_index",
    run_col: str = "run_index",
    intensity_col: str = "ms2_quantity_normalized",
    min_peptides_per_protein: int = 2,
    min_ratio_count: int = 1,
) -> pl.DataFrame:
    """MaxLFQ protein abundance per run.

    Rows with null or non-positive `intensity_col` are dropped. The whole
    per-protein loop (pairwise run ratios, one weakly anchored solve over all
    observed runs) runs in the compiled `_nb_maxlfq_all_proteins`. A run pair
    needs `min_ratio_count` shared peptides for a ratio edge; proteins with
    fewer than `min_peptides_per_protein` peptides are skipped.

    Returns a DataFrame with `protein_col`, `run_col` and Float32 `abundance`.
    """
    df = df.filter(pl.col(intensity_col).is_not_null() & (pl.col(intensity_col) > 0))

    protein_dtype = df.schema[protein_col]
    run_dtype = df.schema[run_col]
    result_schema = {
        protein_col: protein_dtype,
        run_col: run_dtype,
        "abundance": pl.Float32,
    }

    if df.height == 0:
        return pl.DataFrame(schema=result_schema)

    # dense int32 codes, sorted so each protein's rows are contiguous and
    # peptide-ordered (required by _nb_build_L_b); float32 logI/intensity
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
            "abundance": out_abundance,
        },
        schema=result_schema,
    )
