"""Protein picking and group assignment from precursor-filtered evidence."""

from typing import Literal

import polars as pl

from delpi.search.protein_group_mapping import protein_group_mapping
from delpi.search.tda.protein_scoring import (
    DEFAULT_PROTEIN_SCORING,
    ProteinScoringMethod,
    score_proteins,
)

GroupingType = Literal["lead_only", "parsimonious_grouping"]
LIBRARY_COLUMNS = [
    "protein_group",
    "master_protein",
    "library_precursor_q_value",
    "library_peptide_q_value",
    "library_protein_group_q_value",
]


def apply_protein_picker(
    confident_pmsm_df: pl.DataFrame,
    protein_scoring: ProteinScoringMethod = DEFAULT_PROTEIN_SCORING,
    inference_column: str = "peptide_index",
) -> pl.DataFrame:
    """Keep the winning target/decoy membership for each protein index.

    Ties favor targets; unpaired proteins are retained. Scoring uses the
    shared scoring module and all supplied run/precursor evidence.
    """
    pair_df = (
        confident_pmsm_df.select(inference_column, "protein_index", "is_decoy")
        .explode("protein_index", empty_as_null=True)
        .drop_nulls("protein_index")
        .unique()
    )
    if pair_df.is_empty():
        return confident_pmsm_df

    protein_score_df = score_proteins(
        confident_pmsm_df,
        method=protein_scoring,
        group_column="protein_index",
        inference_column=inference_column,
    )
    competition_df = (
        protein_score_df.group_by("protein_index")
        .agg(
            pl.col("score").filter(~pl.col("is_decoy")).max().alias("target_score"),
            pl.col("score").filter(pl.col("is_decoy")).max().alias("decoy_score"),
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
    filtered_protein_df = (
        pair_df.join(competition_df, on="protein_index", how="left")
        .filter(
            (~pl.col("is_decoy") & pl.col("keep_target"))
            | (pl.col("is_decoy") & pl.col("keep_decoy"))
        )
        .group_by([inference_column, "is_decoy"])
        .agg(pl.col("protein_index").unique().sort())
    )
    return (
        confident_pmsm_df.select(pl.exclude("protein_index"))
        .join(filtered_protein_df, on=[inference_column, "is_decoy"], how="left")
        .filter(pl.col("protein_index").is_not_null())
    )


def _infer_membership(
    confident_pmsm_df: pl.DataFrame,
    fasta_id_df: pl.DataFrame,
    grouping_type: GroupingType,
) -> pl.DataFrame:
    # One group per peptide for downstream FDR. Shared peptides select the
    # lowest group_id (greedy-selection order), as in the existing workflow.
    return (
        protein_group_mapping(
            confident_pmsm_df, fasta_id_df, grouping_type=grouping_type
        )
        .sort("group_id")
        .unique("peptide_index", keep="first", maintain_order=True)
        .select("peptide_index", "protein_group", "master_protein")
    )


def assign_protein_groups(
    pmsm_df: pl.DataFrame,
    fasta_id_df: pl.DataFrame,
    q_value_cutoff: float,
    use_protein_picker: bool = True,
    grouping_type: GroupingType = "parsimonious_grouping",
    protein_scoring: ProteinScoringMethod = DEFAULT_PROTEIN_SCORING,
    library_confidence_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach membership using precomputed ``global_precursor_q_value``.

    First-pass grouping uses the existing max(0.05, q_value_cutoff) filter
    and optional protein picking. All run observations remain available to
    the picker. In the second pass, reuse the library's target membership
    and confidence, and freshly group decoys passing q_value_cutoff, without
    picking against the fixed library targets. No q-values are computed here.
    """
    if library_confidence_df is None:
        confident_df = pmsm_df.filter(
            pl.col("global_precursor_q_value") <= max(0.05, q_value_cutoff)
        )
        if use_protein_picker:
            confident_df = apply_protein_picker(confident_df, protein_scoring)
        membership_df = _infer_membership(confident_df, fasta_id_df, grouping_type)
        return pmsm_df.select(pl.exclude("protein_group", "master_protein")).join(
            membership_df, on="peptide_index", how="left", validate="m:1"
        )

    result_df = pmsm_df.select(pl.exclude(*LIBRARY_COLUMNS)).join(
        library_confidence_df.select("precursor_index", *LIBRARY_COLUMNS).with_columns(
            pl.lit(False).alias("is_decoy")
        ),
        on=["precursor_index", "is_decoy"],
        how="left",
        validate="m:1",
    )
    decoy_df = pmsm_df.filter(
        pl.col("is_decoy") & (pl.col("global_precursor_q_value") <= q_value_cutoff)
    )
    if decoy_df.is_empty():
        return result_df
    decoy_membership = _infer_membership(decoy_df, fasta_id_df, grouping_type).rename(
        {
            "protein_group": "_decoy_protein_group",
            "master_protein": "_decoy_master_protein",
        }
    )
    return (
        result_df.join(decoy_membership, on="peptide_index", how="left", validate="m:1")
        .with_columns(
            pl.coalesce("protein_group", "_decoy_protein_group").alias("protein_group"),
            pl.coalesce("master_protein", "_decoy_master_protein").alias(
                "master_protein"
            ),
        )
        .drop("_decoy_protein_group", "_decoy_master_protein")
    )
