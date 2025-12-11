from typing import List
from pathlib import Path

import polars as pl
import numpy as np

from delpi.utils.fdr import calculate_q_value
from delpi.search.protein_group_mapping import protein_group_mapping


class FDRAnalyzer:
    def __init__(
        self,
        q_value_cutoff: float,
        db_dir: Path,
    ):
        self.q_value_cutoff = q_value_cutoff
        self.fasta_id_df = (
            pl.scan_parquet(db_dir / "sequence_df.parquet")
            .select(pl.col("protein_index", "fasta_id"))
            .collect()
        )

    def perform_global_analysis(
        self, pmsm_df: pl.DataFrame, protein_inference: bool = True
    ) -> pl.DataFrame:

        g_pmsm_df = (
            pmsm_df.select(
                pl.col(
                    "precursor_index",
                    "peptidoform_index",
                    "protein_index",
                    "is_decoy",
                    "score",
                )
            )
            .sort(["precursor_index", "score"])
            .unique("precursor_index", keep="last")
        )

        g_pmsm_df = self._update_q_values(
            g_pmsm_df,
            group_keys=["precursor_index"],
            out_column="global_precursor_q_value",
            target_to_decoy_size_ratio=2.0,
        )

        g_pmsm_df = self._update_q_values(
            g_pmsm_df,
            group_keys=["peptidoform_index"],
            out_column="global_peptide_q_value",
            target_to_decoy_size_ratio=2.0,
        )

        if protein_inference:
            # Map protein groups
            confident_pmsm_df = g_pmsm_df.filter(
                pl.col("global_precursor_q_value") <= self.q_value_cutoff
            ).select(pl.col("precursor_index", "is_decoy", "protein_index"))

            pg_df = protein_group_mapping(confident_pmsm_df, self.fasta_id_df)
            g_pmsm_df = g_pmsm_df.select(
                pl.exclude("protein_group", "master_protein")
            ).join(pg_df, on="precursor_index", how="left")

        g_pmsm_df = self._update_q_values(
            g_pmsm_df,
            group_keys=["protein_group", "is_decoy"],
            out_column="global_protein_group_q_value",
            target_to_decoy_size_ratio=1.0,
        )

        pmsm_df = pmsm_df.join(
            g_pmsm_df.select(
                pl.col(
                    "precursor_index",
                    "global_precursor_q_value",
                    "global_peptide_q_value",
                    "global_protein_group_q_value",
                    "protein_group",
                    "master_protein",
                )
            ),
            on="precursor_index",
            how="left",
        )

        return pmsm_df

    def perform_run_specific_analysis(
        self, pmsm_df: pl.DataFrame, protein_inference: bool = True
    ) -> pl.DataFrame:

        pmsm_df = self._update_q_values(
            pmsm_df,
            group_keys=["precursor_index"],
            out_column="precursor_q_value",
            target_to_decoy_size_ratio=2.0,
        )

        pmsm_df = self._update_q_values(
            pmsm_df,
            group_keys=["peptidoform_index"],
            out_column="peptide_q_value",
            target_to_decoy_size_ratio=2.0,
        )

        if protein_inference:
            # Map protein groups
            confident_pmsm_df = pmsm_df.filter(
                pl.col("precursor_q_value") <= self.q_value_cutoff
            ).select(pl.col("precursor_index", "is_decoy", "protein_index"))

            pg_df = protein_group_mapping(confident_pmsm_df, self.fasta_id_df)
            pmsm_df = pmsm_df.select(
                pl.exclude("protein_group", "master_protein")
            ).join(pg_df, on="precursor_index", how="left")

        # Calculate protein group-level Q-values

        pmsm_df = self._update_q_values(
            pmsm_df,
            group_keys=["protein_group", "is_decoy"],
            out_column="protein_group_q_value",
            target_to_decoy_size_ratio=1.0,
        )

        return pmsm_df

    def batch_run_specific_analysis(
        self, pmsm_df: pl.DataFrame, run_key: str = "run_index"
    ) -> pl.DataFrame:

        assert "protein_group" in pmsm_df.columns
        assert "master_protein" in pmsm_df.columns

        dfs = list()
        for run_index, sub_df in pmsm_df.group_by(run_key):
            run_index = run_index[0]
            sub_df = self.perform_run_specific_analysis(sub_df, protein_inference=False)
            dfs.append(sub_df)

        pmsm_df = pl.concat(dfs, how="vertical")

        return pmsm_df

    def _update_q_values(
        self,
        pmsm_df: pl.DataFrame,
        group_keys: List[str],
        out_column: str,
        target_to_decoy_size_ratio: float,
    ) -> pl.DataFrame:

        selected_columns = set(group_keys).union(["score", "is_decoy"])
        df = (
            pmsm_df.select(pl.col(*selected_columns))
            .filter(pl.col(k).is_not_null() for k in group_keys)
            .group_by(group_keys)
            .agg(pl.all().sort_by("score").last())
        )

        df = calculate_q_value(
            df,
            target_to_decoy_size_ratio=target_to_decoy_size_ratio,
            out_column=out_column,
        )

        return pmsm_df.join(
            df.select(pl.col(*group_keys, out_column)), on=group_keys, how="left"
        )


def test():

    fdr_workflow = FDRWorkflow(
        q_value_cutoff=0.01,
        db_dir=Path(r"D:/DelPi_Data/speclib/2021-03-23-reviewed-contam-UP000005640"),
    )
    self = fdr_workflow

    pmsm_df_ = pl.read_parquet("./temp/pmsm_df.parquet")

    pmsm_df = pmsm_df_.filter(pl.col("run_index") == 0)

    output_df = fdr_workflow.perform_run_specific_analysis(
        pmsm_df, protein_inference=True
    )
    output_df.filter(pl.col("is_decoy") == False).filter(
        pl.col("peptide_q_value") <= 0.01
    )["peptidoform_index"].n_unique()

    pmsm_df = pmsm_df_
    output_df = fdr_workflow.perform_global_analysis(pmsm_df)

    output_df.filter(pl.col("is_decoy") == False).filter(
        pl.col("global_peptide_q_value") <= 0.01
    ).filter(pl.col("run_index") == 0)["peptidoform_index"].n_unique()

    output_df2 = fdr_workflow.batch_run_specific_analysis(output_df)

    output_df2.filter(pl.col("is_decoy") == False).filter(
        pl.col("peptide_q_value") <= 0.01
    ).filter(pl.col("run_index") == 0)["peptidoform_index"].n_unique()
