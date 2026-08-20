"""
Second-pass selection utilities.

This module implements the precursor/decoy selection logic used by the
transfer-learning two-pass search:
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from delpi.database.peptide_database import PeptideDatabase
from delpi.chem.modification_param import ModificationParam
from delpi.utils.yaml_file import load_yaml

logger = logging.getLogger(__name__)


def get_decoy_method(db_dir: Path) -> Optional[str]:
    """Read the decoy generation method recorded when the database was built."""
    param_file = Path(db_dir) / "param.yaml"
    return load_yaml(param_file).get("decoy")


def get_var_mod_names(db_dir: Path) -> list[str]:
    param_file = Path(db_dir) / "param.yaml"
    db_params = load_yaml(param_file)

    if "modification" not in db_params:
        return []

    mod = db_params["modification"]

    if "mod_param_set" not in mod:
        return []

    mod_param_set = mod["mod_param_set"]
    mod_params = [ModificationParam(**mods) for mods in mod_param_set]
    return [mod.mod_name for mod in mod_params if mod.fixed == False]


def select_tl_training_pmsms(
    pmsm_df: pl.DataFrame,
    q_value_cutoff: float,
    top_k: int = 3,
) -> pl.DataFrame:
    """Select target PmSM observations used to fine-tune RT/MS2 predictors.

    Only targets that pass *both* the run-specific and the global FDR cutoff
    are eligible.  When the same precursor is confidently identified in
    multiple runs, at most ``top_k`` highest-scoring observations are kept.
    Decoys are never used for fine-tuning.
    """
    df = pmsm_df.filter(
        (pl.col("is_decoy") == False)
        & (pl.col("precursor_q_value") <= q_value_cutoff)
        & (pl.col("global_precursor_q_value") <= q_value_cutoff)
    )

    df = (
        df.sort("score", descending=True)
        .group_by("precursor_index", maintain_order=True)
        .head(top_k)
    )
    return df


def select_second_pass_targets(
    pmsm_df: pl.DataFrame,
    q_value_cutoff: float,
) -> pl.DataFrame:
    """Select unique target precursors confirmed at the experiment level.

    Only targets passing the first-pass ``global_precursor_q_value`` cutoff
    are eligible for the refined (second-pass) target-decoy library.  Targets
    that fail the first-pass global cutoff are not searched in the second
    pass.
    """
    df = (
        pmsm_df.filter(
            (pl.col("is_decoy") == False)
            & (pl.col("global_precursor_q_value") <= q_value_cutoff)
        )
        .sort(["precursor_index", "score"])
        .unique("precursor_index", keep="last")
    )
    return df


def select_paired_decoys(
    db_dir: Path,
    target_df: pl.DataFrame,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Pair each target precursor with one decoy precursor generated from it.

    ``target_df`` must contain ONLY target rows (``is_decoy == False``) and
    already carry ``precursor_index``, ``peptide_index``, ``peptidoform_index``
    and ``precursor_charge`` (as produced by ``PeptideDatabase.join`` /
    ``select_second_pass_targets``). The paired decoy must share the same
    ``precursor_charge`` and ``var_mod_profile`` (per-variable-modification
    occurrence counts) as the target. Ties are broken deterministically via
    ``random_seed``.

    Databases built before pair-preserving decoy generation lack a
    ``target_peptide_index`` column on ``peptide_df``; for those, pairing
    falls back to :func:`_select_paired_decoys_by_nearest_mz`.
    """
    db_dir = Path(db_dir)

    if "target_peptide_index" not in (
        pl.scan_parquet(db_dir / "peptide_df.parquet").collect_schema()
    ):
        paired_df = _select_paired_decoys_by_nearest_mz(
            db_dir, target_df, random_seed=random_seed
        )
    else:
        paired_df = _select_paired_decoys_by_sequence(
            db_dir, target_df, random_seed=random_seed
        )

    n_missing = paired_df["decoy_precursor_index"].null_count()
    if n_missing:
        logger.debug(
            f"Could not find a paired decoy for {n_missing}/{len(paired_df)} "
            "target precursors; they will be excluded from the refined library."
        )

    return paired_df


def _select_paired_decoys_by_sequence(
    db_dir: Path,
    target_df: pl.DataFrame,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Pair via ``peptide_df.target_peptide_index`` (see :func:`select_paired_decoys`)."""
    var_mod_names = get_var_mod_names(db_dir)

    def _var_mod_profile_expr() -> pl.Expr:
        # pl.struct requires at least one field; when there are no variable
        # modifications, fall back to a constant field so every row shares
        # the same (trivial) profile and the charge-only match still works.
        mod_count_exprs = [
            pl.col("mods")
            .str.split(";")
            .fill_null(pl.lit([], dtype=pl.List(pl.String)))
            .list.count_matches(mod_name)
            .cast(pl.UInt8)
            .alias(f"n_{mod_name.lower()}")
            for mod_name in var_mod_names
        ] or [pl.lit(0, dtype=pl.UInt8).alias("_no_var_mods")]
        return pl.struct(mod_count_exprs).alias("var_mod_profile")

    modification_df = pl.scan_parquet(db_dir / "modification_df.parquet").select(
        "peptidoform_index", "peptide_index", "mods"
    )
    precursor_df = pl.scan_parquet(db_dir / "precursor_df.parquet").select(
        "peptidoform_index", "precursor_index", "precursor_charge"
    )

    # each target's own var-mod profile
    target_df = (
        target_df.lazy()
        .join(
            modification_df.select("peptidoform_index", "mods"),
            on="peptidoform_index",
            how="left",
        )
        .with_columns(_var_mod_profile_expr())
        .collect()
    )

    # target_peptide_index -> decoy_peptide_index map, restricted to peptides
    # that actually appear in target_df for efficiency
    target_peptides = target_df.lazy().select(pl.col("peptide_index").unique())
    decoy_map_df = (
        pl.scan_parquet(db_dir / "peptide_df.parquet")
        .filter(pl.col("is_decoy"))
        .select(
            pl.col("peptide_index").alias("decoy_peptide_index"), "target_peptide_index"
        )
        .explode("target_peptide_index")
        .join(
            target_peptides,
            left_on="target_peptide_index",
            right_on="peptide_index",
            how="inner",
        )
    )

    # candidate decoy pool: every decoy precursor's own charge + var-mod
    # profile, keyed by the target peptide it was generated from
    candidate_df = (
        precursor_df.join(modification_df, on="peptidoform_index", how="left")
        .with_columns(_var_mod_profile_expr())
        .join(
            decoy_map_df,
            left_on="peptide_index",
            right_on="decoy_peptide_index",
            how="inner",
        )
        .select(
            pl.col("target_peptide_index").alias("peptide_index"),
            pl.col("precursor_index").alias("decoy_precursor_index"),
            "precursor_charge",
            "var_mod_profile",
        )
    )

    paired_df = (
        target_df.lazy()
        .join(
            candidate_df,
            on=["peptide_index", "precursor_charge", "var_mod_profile"],
            how="left",
        )
        .collect()
    )

    return (
        paired_df.with_columns(
            pl.int_range(pl.len()).shuffle(seed=random_seed).alias("_random_order")
        )
        .sort("_random_order")
        .unique(subset=["precursor_index"], keep="first", maintain_order=True)
        .drop("_random_order", "var_mod_profile", "mods")
    )


def _select_paired_decoys_by_nearest_mz(
    db_dir: Path,
    target_df: pl.DataFrame,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Pair each target with the nearest-m/z decoy of same length/charge.

    Legacy-database fallback used when ``peptide_df`` has no
    ``target_peptide_index`` column to pair by sequence lineage: candidates
    are restricted to the same ``sequence_length`` and ``precursor_charge``
    only, so every target still gets a (best-effort) decoy count-matched to
    it. Ties in m/z distance are broken deterministically via
    ``random_seed``.
    """
    peptide_df = pl.read_parquet(db_dir / "peptide_df.parquet").select(
        pl.col("peptide_index", "is_decoy", "sequence_length")
    )
    modification_df = pl.read_parquet(db_dir / "modification_df.parquet").select(
        pl.col("peptidoform_index", "peptide_index")
    )
    precursor_df = pl.read_parquet(db_dir / "precursor_df.parquet").select(
        pl.col(
            "precursor_index", "peptidoform_index", "precursor_charge", "precursor_mz"
        )
    )

    decoy_candidates = (
        precursor_df.join(modification_df, on="peptidoform_index", how="left")
        .join(peptide_df.filter(pl.col("is_decoy")), on="peptide_index", how="inner")
        .select(
            pl.col("precursor_index").alias("decoy_precursor_index"),
            pl.col("precursor_charge"),
            pl.col("precursor_mz").alias("decoy_precursor_mz"),
            pl.col("sequence_length"),
        )
    )

    target_meta_df = target_df.select(
        pl.exclude("precursor_mz") if "precursor_mz" in target_df.columns else pl.all()
    ).join(
        precursor_df.select(pl.col("precursor_index", "precursor_mz")),
        on="precursor_index",
        how="left",
    )

    rng = np.random.default_rng(random_seed)
    result_frames = []
    for (charge, seq_len), group in target_meta_df.group_by(
        ["precursor_charge", "sequence_length"]
    ):
        cand = decoy_candidates.filter(
            (pl.col("precursor_charge") == charge)
            & (pl.col("sequence_length") == seq_len)
        ).sort("decoy_precursor_mz")

        if cand.is_empty():
            result_frames.append(
                group.with_columns(
                    pl.lit(None, dtype=pl.UInt32).alias("decoy_precursor_index")
                )
            )
            continue

        decoy_mz_arr = cand["decoy_precursor_mz"].to_numpy()
        decoy_idx_arr = cand["decoy_precursor_index"].to_numpy()
        target_mz_arr = group["precursor_mz"].to_numpy()

        pos = np.searchsorted(decoy_mz_arr, target_mz_arr)
        pos_left = np.clip(pos - 1, 0, len(decoy_mz_arr) - 1)
        pos_right = np.clip(pos, 0, len(decoy_mz_arr) - 1)
        diff_left = np.abs(decoy_mz_arr[pos_left] - target_mz_arr)
        diff_right = np.abs(decoy_mz_arr[pos_right] - target_mz_arr)

        tie = diff_right == diff_left
        tie_pick_right = rng.random(len(target_mz_arr)) < 0.5
        pick_right = np.where(tie, tie_pick_right, diff_right < diff_left)
        chosen_pos = np.where(pick_right, pos_right, pos_left)
        chosen_decoy_idx = decoy_idx_arr[chosen_pos]

        result_frames.append(
            group.with_columns(
                pl.Series("decoy_precursor_index", chosen_decoy_idx, dtype=pl.UInt32)
            )
        )

    paired_df = pl.concat(result_frames, how="vertical").select(
        pl.col("precursor_index", "decoy_precursor_index")
    )

    return target_df.join(paired_df, on="precursor_index", how="left")

