from typing import List, Literal, Optional
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
        grouping_type: Literal[
            "lead_only", "parsimonious_grouping"
        ] = "parsimonious_grouping",
    ):
        self.q_value_cutoff = q_value_cutoff
        self.use_protein_picker = use_protein_picker
        self.grouping_type = grouping_type
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
        library_confidence_df: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Compute global precursor/peptide (and, optionally, protein-group)
        q-values.

        When ``library_confidence_df`` is provided (second pass of the
        two-pass MBR search), protein-group membership is *not* recomputed
        from scratch for targets: target rows reuse the first pass's
        ``protein_group``/``master_protein`` and ``library_*_q_value``
        columns (joined by ``precursor_index`` from
        ``library_confidence.parquet``; decoy rows get ``null`` for the
        library q-values, since they weren't part of the first pass), while
        decoy rows are freshly grouped here so they can compete for a
        protein-group q-value. This takes precedence over
        ``protein_inference``.
        """

        g_pmsm_df = (
            pmsm_df.select(
                pl.col(
                    "precursor_index",
                    "peptide_index",
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

        join_columns = [
            "precursor_index",
            "global_precursor_q_value",
            "global_peptide_q_value",
        ]

        if library_confidence_df is not None:
            # Second pass: reuse the first pass's target protein-group
            # membership and library q-values instead of recomputing them
            # (decoy rows get null for the library q-values, filled in below
            # for protein_group/master_protein only).
            g_pmsm_df = g_pmsm_df.join(
                library_confidence_df.select(
                    pl.col(
                        "precursor_index",
                        "protein_group",
                        "master_protein",
                        "library_precursor_q_value",
                        "library_peptide_q_value",
                        "library_protein_group_q_value",
                    )
                ),
                on="precursor_index",
                how="left",
            )

            # Freshly group only the confident decoys so they have a
            # protein_group to compete against the (reused) target groups.
            decoy_confident_df = g_pmsm_df.filter(
                pl.col("is_decoy")
                & (pl.col("global_precursor_q_value") <= self.q_value_cutoff)
            ).select(pl.col("peptide_index", "is_decoy", "protein_index", "score"))

            if not decoy_confident_df.is_empty():
                decoy_pg_df = protein_group_mapping(
                    decoy_confident_df,
                    self.fasta_id_df,
                    grouping_type=self.grouping_type,
                )
                decoy_pg_df_dedup = (
                    decoy_pg_df.sort("group_id")
                    .unique("peptide_index", keep="first", maintain_order=True)
                    .select(
                        pl.col("peptide_index"),
                        pl.col("protein_group").alias("_decoy_protein_group"),
                        pl.col("master_protein").alias("_decoy_master_protein"),
                    )
                )
                g_pmsm_df = (
                    g_pmsm_df.join(decoy_pg_df_dedup, on="peptide_index", how="left")
                    .with_columns(
                        pl.coalesce("protein_group", "_decoy_protein_group").alias(
                            "protein_group"
                        ),
                        pl.coalesce("master_protein", "_decoy_master_protein").alias(
                            "master_protein"
                        ),
                    )
                    .drop("_decoy_protein_group", "_decoy_master_protein")
                )

            g_pmsm_df = self._update_q_values(
                g_pmsm_df,
                group_keys=["protein_group", "is_decoy"],
                out_column="global_protein_group_q_value",
                target_to_decoy_size_ratio=1.0,
            )
            join_columns += [
                "global_protein_group_q_value",
                "protein_group",
                "master_protein",
                "library_precursor_q_value",
                "library_peptide_q_value",
                "library_protein_group_q_value",
            ]

        elif protein_inference:
            # Map protein groups
            confident_pmsm_df = g_pmsm_df.filter(
                pl.col("global_precursor_q_value") <= self.q_value_cutoff
            ).select(pl.col("peptide_index", "is_decoy", "protein_index", "score"))

            if self.use_protein_picker:
                confident_pmsm_df = self._apply_protein_picker(confident_pmsm_df)

            pg_df = protein_group_mapping(
                confident_pmsm_df,
                self.fasta_id_df,
                grouping_type=self.grouping_type,
            )
            # protein_group_mapping returns one row per (peptide_index, group_id)
            # edge (many-to-many). For precursor-level downstream FDR we need
            # exactly one protein_group per peptide_index. Deduplicate to the
            # lowest group_id (greedy-selection order) so the choice is
            # deterministic and independent of DataFrame row order.
            pg_df_dedup = (
                pg_df.sort("group_id")
                .unique("peptide_index", keep="first", maintain_order=True)
                .drop("group_id")
            )
            g_pmsm_df = g_pmsm_df.select(
                pl.exclude("protein_group", "master_protein")
            ).join(pg_df_dedup, on="peptide_index", how="left")

            g_pmsm_df = self._update_q_values(
                g_pmsm_df,
                group_keys=["protein_group", "is_decoy"],
                out_column="global_protein_group_q_value",
                target_to_decoy_size_ratio=1.0,
            )
            join_columns += [
                "global_protein_group_q_value",
                "protein_group",
                "master_protein",
            ]

        pmsm_df = pmsm_df.join(
            g_pmsm_df.select(pl.col(*join_columns)),
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
            ).select(pl.col("peptide_index", "is_decoy", "protein_index", "score"))

            if self.use_protein_picker:
                confident_pmsm_df = self._apply_protein_picker(confident_pmsm_df)

            pg_df = protein_group_mapping(
                confident_pmsm_df,
                self.fasta_id_df,
                grouping_type=self.grouping_type,
            )
            # Deduplicate many-to-many edges to one group per peptide_index
            # (lowest group_id = greedy-selection order).
            pg_df_dedup = (
                pg_df.sort("group_id")
                .unique("peptide_index", keep="first", maintain_order=True)
                .drop("group_id")
            )
            pmsm_df = pmsm_df.select(
                pl.exclude("protein_group", "master_protein")
            ).join(pg_df_dedup, on="peptide_index", how="left")

        if "protein_group" in pmsm_df.columns:
            # Calculate protein group-level Q-values
            pmsm_df = self._update_q_values(
                pmsm_df,
                group_keys=["protein_group", "is_decoy"],
                out_column="protein_group_q_value",
                target_to_decoy_size_ratio=1.0,
            )

        return pmsm_df

    def _apply_protein_picker(
        self,
        confident_pmsm_df: pl.DataFrame,
        inference_column: str = "peptide_index",
    ) -> pl.DataFrame:
        pair_df = (
            confident_pmsm_df.select(
                pl.col(inference_column, "protein_index", "is_decoy", "score")
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

        filtered_protein_df = filtered_pair_df.group_by(inference_column).agg(
            pl.col("protein_index")
        )

        return (
            confident_pmsm_df.select(pl.exclude("protein_index"))
            .join(filtered_protein_df, on=inference_column, how="left")
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
