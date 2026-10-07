"""Shared protein and protein-group scoring from PmSM evidence."""

import logging
from typing import Literal, get_args

import numpy as np
import polars as pl

ProteinScoringMethod = Literal[
    "best_peptide_per_protein", "top_two_combined", "discriminative_rescoring"
]
DEFAULT_PROTEIN_SCORING: ProteinScoringMethod = "top_two_combined"
PROTEIN_FEATURE_COLUMNS = (
    "num_runs",
    "num_peptides_per_run",
    "max_num_peptides_per_run",
    "num_precursors",
    "num_peptides",
    "max_score",
    "min_score",
    "mean_score",
    "std_score",
)

logger = logging.getLogger(__name__)


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
    columns += [
        name
        for name in ("run_index", "precursor_index")
        if name in pmsm_df.columns and name not in columns
    ]
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


def _aggregate_features(
    evidence_df: pl.DataFrame, group_column: str, inference_column: str
) -> pl.DataFrame:
    keys = [group_column, "is_decoy"]
    if "run_index" not in evidence_df.columns:
        evidence_df = evidence_df.with_columns(pl.lit(0).alias("run_index"))
    precursor_column = (
        "precursor_index"
        if "precursor_index" in evidence_df.columns
        else inference_column
    )

    # Average over observed runs, not over PmSM rows (which would overweight
    # runs with more observations). Keep each protein's counts independent.
    run_counts = (
        evidence_df.group_by([*keys, "run_index"])
        .agg(pl.col(inference_column).n_unique().alias("_num_peptides"))
        .group_by(keys)
        .agg(
            pl.len().alias("num_runs"),
            pl.col("_num_peptides").mean().alias("num_peptides_per_run"),
            pl.col("_num_peptides").max().alias("max_num_peptides_per_run"),
        )
    )
    return (
        evidence_df.group_by(keys)
        .agg(
            pl.col(precursor_column).n_unique().alias("num_precursors"),
            pl.col(inference_column).n_unique().alias("num_peptides"),
            pl.col("score").max().alias("max_score"),
            pl.col("score").min().alias("min_score"),
            pl.col("score").mean().alias("mean_score"),
            pl.col("score").std().alias("std_score"),
        )
        .join(run_counts, on=keys, how="left")
        .select(*keys, *PROTEIN_FEATURE_COLUMNS)
        .sort(keys)
    )


def extract_protein_features(
    pmsm_df: pl.DataFrame,
    group_column: str = "protein_group",
    inference_column: str = "peptide_index",
) -> pl.DataFrame:
    """Extract PoC count and score features per (protein/group, is_decoy).

    Use ``group_column="protein_index"`` for individual proteins; list-valued
    memberships are exploded. ``protein_group`` is the existing group label.
    Required columns are the group key, ``is_decoy``, ``score`` and the
    inference key (stripped peptide by default). Missing ``run_index`` means
    one run; missing ``precursor_index`` uses peptide counts. Null keys and
    non-finite scores are excluded. Singleton score standard deviations are
    null, and are imputed only when training the discriminative model.
    """
    return _aggregate_features(
        _prepare_evidence(pmsm_df, group_column, inference_column),
        group_column,
        inference_column,
    )


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
    random_state: int = 42,
) -> pl.DataFrame:
    """Return group key, ``is_decoy`` and a higher-is-better Float32 ``score``.

    The peptide methods use the maximum PmSM score per distinct stripped
    peptide, then take the best peptide or sum the best two. A singleton
    keeps its one peptide score.

    Discriminative rescoring uses two-fold stratified out-of-fold logistic
    regression, with target probability as the score. Feature scaling is
    fitted on each training fold only. Sorting entity keys before splitting
    makes results reproducible across input row orders. Each call fits fresh
    models for its evidence (global, run-specific, or pre-picker proteins).
    Fewer than two targets or two decoys falls back to top_two_combined.
    """
    validate_protein_scoring(method)
    evidence_df = _prepare_evidence(pmsm_df, group_column, inference_column)
    if method != "discriminative_rescoring" or evidence_df.is_empty():
        return _peptide_scores(
            evidence_df,
            group_column,
            inference_column,
            top_n=1 if method == "best_peptide_per_protein" else 2,
        )

    feature_df = _aggregate_features(evidence_df, group_column, inference_column)
    y = feature_df["is_decoy"].cast(pl.Int8).to_numpy()
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        logger.warning(
            "Protein discriminative_rescoring requires at least two targets and "
            "two decoys; got %d targets and %d decoys for %s. "
            "Falling back to top_two_combined.",
            counts[0],
            counts[1],
            group_column,
        )
        return _peptide_scores(evidence_df, group_column, inference_column, top_n=2)

    # Keep sklearn out of the default heuristic scoring path.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X = feature_df.select(PROTEIN_FEATURE_COLUMNS).to_numpy().astype(np.float64)
    X = np.nan_to_num(X, nan=-10.0, posinf=-10.0, neginf=-10.0)
    scores = np.empty(len(feature_df), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
    for train_idx, score_idx in splitter.split(X, y):
        model = LogisticRegression(random_state=random_state)
        model.fit(X[train_idx], y[train_idx])
        # y=0 is target: equivalent to 1 - P(decoy), without cancellation.
        scores[score_idx] = model.predict_proba(X[score_idx])[:, 0]

    return feature_df.select(group_column, "is_decoy").with_columns(
        pl.Series("score", scores, dtype=pl.Float32)
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

    Global scoring sees all run observations. Each run is scored separately,
    fitting its own ML models when requested. Scores are joined by group and
    target/decoy status (and run for run-specific scores); ungrouped rows keep
    null scores. This stage performs no inference or FDR estimation.
    """
    validate_protein_scoring(method)
    if pmsm_df.is_empty():
        return pmsm_df.with_columns(
            pl.lit(None, dtype=pl.Float32).alias(global_score_column),
            pl.lit(None, dtype=pl.Float32).alias(run_score_column),
        )

    keys = ["protein_group", "is_decoy"]
    evidence_df = pmsm_df.with_columns(
        pl.col(score_column).alias("score"),
        pl.col(run_key).alias("run_index"),
    )
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
