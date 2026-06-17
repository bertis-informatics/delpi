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
        use_protein_picker: bool = True,
    ):
        self.q_value_cutoff = q_value_cutoff
        self.use_protein_picker = use_protein_picker
        self.fasta_id_df = (
            pl.scan_parquet(db_dir / "sequence_df.parquet")
            .select(pl.col("protein_index", "fasta_id"))
            .collect()
        )

    def perform_global_analysis(
        self,
        pmsm_df: pl.DataFrame,
        protein_inference: bool = True,
        target_to_decoy_size_ratio: float = 1.0,
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
            .sort(["peptidoform_index", "precursor_index", "score"])
            .unique("precursor_index", keep="last")
        )

        g_pmsm_df = self._update_q_values(
            g_pmsm_df,
            group_keys=["precursor_index"],
            out_column="global_precursor_q_value",
            target_to_decoy_size_ratio=target_to_decoy_size_ratio,
        )

        g_pmsm_df = self._update_q_values(
            g_pmsm_df,
            group_keys=["peptidoform_index"],
            out_column="global_peptide_q_value",
            target_to_decoy_size_ratio=target_to_decoy_size_ratio,
        )

        if protein_inference:
            # Map protein groups
            confident_pmsm_df = g_pmsm_df.filter(
                pl.col("global_precursor_q_value") <= self.q_value_cutoff
            ).select(pl.col("precursor_index", "is_decoy", "protein_index", "score"))

            if self.use_protein_picker:
                confident_pmsm_df = self._apply_protein_picker(confident_pmsm_df)

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
        self,
        pmsm_df: pl.DataFrame,
        protein_inference: bool = True,
        target_to_decoy_size_ratio: float = 1.0,
    ) -> pl.DataFrame:

        pmsm_df = self._update_q_values(
            pmsm_df,
            group_keys=["precursor_index"],
            out_column="precursor_q_value",
            target_to_decoy_size_ratio=target_to_decoy_size_ratio,
        )

        pmsm_df = self._update_q_values(
            pmsm_df,
            group_keys=["peptidoform_index"],
            out_column="peptide_q_value",
            target_to_decoy_size_ratio=target_to_decoy_size_ratio,
        )

        if protein_inference:
            # Map protein groups
            confident_pmsm_df = pmsm_df.filter(
                pl.col("precursor_q_value") <= self.q_value_cutoff
            ).select(pl.col("precursor_index", "is_decoy", "protein_index", "score"))

            if self.use_protein_picker:
                confident_pmsm_df = self._apply_protein_picker(confident_pmsm_df)

            pg_df = protein_group_mapping(confident_pmsm_df, self.fasta_id_df)
            pmsm_df = pmsm_df.select(
                pl.exclude("protein_group", "master_protein")
            ).join(pg_df, on="precursor_index", how="left")

        if "protein_group" in pmsm_df.columns:
            # Calculate protein group-level Q-values
            pmsm_df = self._update_q_values(
                pmsm_df,
                group_keys=["protein_group", "is_decoy"],
                out_column="protein_group_q_value",
                target_to_decoy_size_ratio=1.0,
            )

        return pmsm_df

    def _apply_protein_picker(self, confident_pmsm_df: pl.DataFrame) -> pl.DataFrame:
        pair_df = (
            confident_pmsm_df.select(
                pl.col("precursor_index", "protein_index", "is_decoy", "score")
            )
            .explode("protein_index")
            .drop_nulls("protein_index")
        )

        if pair_df.is_empty():
            return confident_pmsm_df

        protein_score_df = pair_df.group_by(["protein_index", "is_decoy"]).agg(
            pl.col("score").max().alias("protein_score")
        )

        competition_df = (
            protein_score_df.group_by("protein_index")
            .agg(
                pl.col("protein_score")
                .filter(~pl.col("is_decoy"))
                .max()
                .alias("target_score"),
                pl.col("protein_score")
                .filter(pl.col("is_decoy"))
                .max()
                .alias("decoy_score"),
            )
            .with_columns(
                (
                    pl.col("decoy_score").is_null()
                    | (
                        pl.col("target_score").is_not_null()
                        & (pl.col("target_score") >= pl.col("decoy_score"))
                    )
                ).alias("keep_target"),
                (
                    pl.col("target_score").is_null()
                    | (
                        pl.col("decoy_score").is_not_null()
                        & (pl.col("decoy_score") > pl.col("target_score"))
                    )
                ).alias("keep_decoy"),
            )
        )

        filtered_pair_df = pair_df.join(
            competition_df.select(pl.col("protein_index", "keep_target", "keep_decoy")),
            on="protein_index",
            how="left",
        ).filter(
            ((pl.col("is_decoy") == False) & pl.col("keep_target"))
            | ((pl.col("is_decoy") == True) & pl.col("keep_decoy"))
        )

        filtered_protein_df = filtered_pair_df.group_by("precursor_index").agg(
            pl.col("protein_index")
        )

        return (
            confident_pmsm_df.select(pl.exclude("protein_index"))
            .join(filtered_protein_df, on="precursor_index", how="left")
            .filter(pl.col("protein_index").is_not_null())
        )

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

    def add_fasta_id_column(self, pmsm_df):
        # add fasta ID columns
        tmp_df = (
            pmsm_df.select(pl.col("precursor_index", "protein_index"))
            .unique(["precursor_index"], keep="first")
            .explode("protein_index")
            .join(self.fasta_id_df, on="protein_index", how="left")
            .group_by("precursor_index")
            .agg(pl.col("fasta_id"))
            .with_columns(pl.col("fasta_id").list.sort().list.join(";"))
        )

        return pmsm_df.select(pl.exclude("fasta_id")).join(
            tmp_df, on="precursor_index", how="left"
        )
