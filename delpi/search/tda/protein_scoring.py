"""Shared protein and protein-group scoring from PmSM evidence."""

from typing import Literal, get_args

import polars as pl

ProteinScoringMethod = Literal["best_peptide_per_protein", "top_two_combined"]
DEFAULT_PROTEIN_SCORING: ProteinScoringMethod = "best_peptide_per_protein"
_TOP_N = {"best_peptide_per_protein": 1, "top_two_combined": 2}


def validate_protein_scoring(method: str) -> ProteinScoringMethod:
    if method not in get_args(ProteinScoringMethod):
        raise ValueError(
            f"Unsupported protein_scoring {method!r}. "
            f"Supported methods: {', '.join(get_args(ProteinScoringMethod))}"
        )
    return method


def _prepare_evidence(
    pmsm_df: pl.DataFrame, group_column: str, inference_column: str
) -> pl.DataFrame:
    columns = [group_column, "is_decoy", inference_column, "score"]
    evidence_df = pmsm_df.select(columns)
    if isinstance(evidence_df.schema[group_column], pl.List):
        # A repeated protein ID in a peptide's membership must not duplicate
        # its evidence. Shared peptides still contribute to each protein.
        evidence_df = evidence_df.with_columns(
            pl.col(group_column).list.unique()
        ).explode(group_column, empty_as_null=True)
    return evidence_df.drop_nulls(
        [group_column, "is_decoy", inference_column, "score"]
    ).filter(pl.col("score").is_finite())


def _peptide_scores(
    evidence_df: pl.DataFrame,
    group_column: str,
    inference_column: str,
    top_n: int,
) -> pl.DataFrame:
    keys = [group_column, "is_decoy"]
    return (
        evidence_df.group_by([*keys, inference_column])
        .agg(pl.col("score").max())
        .group_by(keys)
        .agg(pl.col("score").sort(descending=True).head(top_n).sum().cast(pl.Float32))
        .sort(keys)
    )


def score_proteins(
    pmsm_df: pl.DataFrame,
    method: ProteinScoringMethod = DEFAULT_PROTEIN_SCORING,
    group_column: str = "protein_group",
    inference_column: str = "peptide_index",
) -> pl.DataFrame:
    """Return group key, ``is_decoy`` and a higher-is-better Float32 ``score``.

    Each distinct stripped peptide takes its maximum PmSM score; a group scores
    its best peptide (``best_peptide_per_protein``) or the sum of its best two
    (``top_two_combined``, a singleton keeps its one score).
    """
    validate_protein_scoring(method)
    return _peptide_scores(
        _prepare_evidence(pmsm_df, group_column, inference_column),
        group_column,
        inference_column,
        top_n=_TOP_N[method],
    )


def score_protein_groups(
    pmsm_df: pl.DataFrame,
    method: ProteinScoringMethod = DEFAULT_PROTEIN_SCORING,
    run_key: str = "run_index",
    score_column: str = "score",
    global_score_column: str = "global_protein_group_score",
    run_score_column: str = "protein_group_score",
) -> pl.DataFrame:
    """Attach global and per-run protein-group scores to grouped PmSMs.

    Global scoring sees all run observations; each run is scored separately.
    Scores are joined by group and target/decoy status (and run for
    run-specific scores); ungrouped rows keep null scores. This stage performs
    no inference or FDR estimation.
    """
    validate_protein_scoring(method)
    if pmsm_df.is_empty():
        return pmsm_df.with_columns(
            pl.lit(None, dtype=pl.Float32).alias(global_score_column),
            pl.lit(None, dtype=pl.Float32).alias(run_score_column),
        )

    keys = ["protein_group", "is_decoy"]
    evidence_df = pmsm_df.with_columns(pl.col(score_column).alias("score"))
    global_scores = score_proteins(evidence_df, method=method).rename(
        {"score": global_score_column}
    )
    run_scores = []
    for (run_index,), run_df in evidence_df.group_by(run_key, maintain_order=True):
        run_scores.append(
            score_proteins(run_df, method=method).with_columns(
                pl.lit(run_index, dtype=pmsm_df.schema[run_key]).alias(run_key),
            )
        )
    run_scores_df = pl.concat(run_scores).rename({"score": run_score_column})
    return (
        pmsm_df.select(pl.exclude(global_score_column, run_score_column))
        .join(global_scores, on=keys, how="left", validate="m:1")
        .join(run_scores_df, on=[run_key, *keys], how="left", validate="m:1")
    )
