import sys
from pathlib import Path

import polars as pl
from delpi.search.dia.max_lfq import maxlfq


def main(pmsm_path: str):
    """Run MaxLFQ protein quantification on PMSM results.
    
    Args:
        pmsm_path: Path to the PMSM results TSV file
    """
    pmsm_path = Path(pmsm_path)
    
    pmsm_df = (
        pl.scan_csv(pmsm_path, separator="\t")
        .filter(pl.col("is_decoy") == False)
        .filter(pl.col("global_protein_group_q_value") <= 0.01)
        .collect()
    )

    ## collapse_precursors_to_peptides
    df = (
        pmsm_df.filter(pl.col("ms2_area").is_not_null() & (pl.col("ms2_area") > 0))
        .group_by(["protein_group", "peptide_index", "run_name"])
        .agg(
            pl.col("ms2_area").median().alias("peptide_abundance"),
        )
    )

    pg_quant_df = maxlfq(df, run_col="run_name", min_peptides_per_protein=1)
    output_path = pmsm_path.parent / "protein_group_maxlfq_results.tsv"
    pg_quant_df.write_csv(output_path, separator="\t")
    print(f"MaxLFQ results saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_maxlfq.py <pmsm_path>")
        sys.exit(1)
    
    main(sys.argv[1])