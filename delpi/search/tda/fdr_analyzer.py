"""Global and run-specific FDR estimation from precomputed scores."""

import polars as pl

from delpi.utils.fdr import calculate_q_value


class FDRAnalyzer:
    """q-values from precomputed scores (higher is better).

    No protein inference here: protein-group FDR needs ``protein_group`` and
    its score supplied by the caller.
    """

    def perform_global_analysis(
        self,
        pmsm_df: pl.DataFrame,
        score_column: str = "score",
        protein_group_score_column: str = "protein_group_score",
        target_to_decoy_size_ratio: float = 1.0,
    ) -> pl.DataFrame:
        """Global precursor, peptide and optional protein-group q-values.

        Precursors/peptides use their best PmSM score across runs; protein
        groups use `protein_group_score_column` (e.g.
        ``global_protein_group_score``). Existing q-value columns are replaced.
        """
        pmsm_df = self.calculate_q_value(
            pmsm_df,
            group_keys=["precursor_index"],
            score_column=score_column,
            out_column="global_precursor_q_value",
            target_to_decoy_size_ratio=target_to_decoy_size_ratio,
        )
        pmsm_df = self.calculate_q_value(
            pmsm_df,
            group_keys=["peptidoform_index"],
            score_column=score_column,
            out_column="global_peptide_q_value",
            target_to_decoy_size_ratio=target_to_decoy_size_ratio,
        )
        if "protein_group" in pmsm_df.columns:
            pmsm_df = self.calculate_q_value(
                pmsm_df,
                group_keys=["protein_group", "is_decoy"],
                score_column=protein_group_score_column,
                out_column="global_protein_group_q_value",
                target_to_decoy_size_ratio=1.0,
            )
        return pmsm_df

    def _perform_run_specific_analysis(
        self,
        pmsm_df: pl.DataFrame,
        score_column: str = "score",
        protein_group_score_column: str = "protein_group_score",
        target_to_decoy_size_ratio: float = 1.0,
    ) -> pl.DataFrame:
        """Calculate q-values for a single run, using its precomputed scores."""
        pmsm_df = self.calculate_q_value(
            pmsm_df,
            group_keys=["precursor_index"],
            score_column=score_column,
            out_column="precursor_q_value",
            target_to_decoy_size_ratio=target_to_decoy_size_ratio,
        )
        pmsm_df = self.calculate_q_value(
            pmsm_df,
            group_keys=["peptidoform_index"],
            score_column=score_column,
            out_column="peptide_q_value",
            target_to_decoy_size_ratio=target_to_decoy_size_ratio,
        )
        if "protein_group" in pmsm_df.columns:
            pmsm_df = self.calculate_q_value(
                pmsm_df,
                group_keys=["protein_group", "is_decoy"],
                score_column=protein_group_score_column,
                out_column="protein_group_q_value",
                target_to_decoy_size_ratio=target_to_decoy_size_ratio,
            )
        return pmsm_df

    def perform_run_specific_analysis(
        self,
        pmsm_df: pl.DataFrame,
        run_key: str = "run_index",
        score_column: str = "score",
        protein_group_score_column: str = "protein_group_score",
        target_to_decoy_size_ratio: float = 1.0,
    ) -> pl.DataFrame:
        """Analyze each run independently and concatenate the annotated rows."""
        if pmsm_df.is_empty():
            return self._perform_run_specific_analysis(
                pmsm_df,
                score_column,
                protein_group_score_column,
                target_to_decoy_size_ratio,
            )
        return pl.concat(
            [
                self._perform_run_specific_analysis(
                    sub_df,
                    score_column,
                    protein_group_score_column,
                    target_to_decoy_size_ratio,
                )
                for _, sub_df in pmsm_df.group_by(run_key, maintain_order=True)
            ],
            how="vertical",
        )

    @staticmethod
    def calculate_q_value(
        pmsm_df: pl.DataFrame,
        group_keys: list[str],
        score_column: str,
        out_column: str,
        target_to_decoy_size_ratio: float,
    ) -> pl.DataFrame:
        """Per-group q-values joined back onto every row.

        Each group contributes its highest finite ``score_column`` value and
        its ``is_decoy``; rows with null keys or scores are ignored and get a
        null q-value if their group has no valid row. ``out_column`` is
        replaced. The frame defines the scope (all runs or one run).
        """
        selected_columns = list(dict.fromkeys([*group_keys, score_column, "is_decoy"]))
        score_df = (
            pmsm_df.select(selected_columns)
            .drop_nulls(selected_columns)
            .filter(pl.col(score_column).is_finite())
            .group_by(group_keys)
            .agg(pl.all().sort_by(score_column).last())
        )
        if "protein_group" in group_keys:
            # keep the protein-scoring key order after collapsing member rows
            score_df = score_df.sort(group_keys)
        score_df = calculate_q_value(
            score_df,
            score_column=score_column,
            target_to_decoy_size_ratio=target_to_decoy_size_ratio,
            out_column=out_column,
        )
        return pmsm_df.select(pl.exclude(out_column)).join(
            score_df.select(*group_keys, out_column),
            on=group_keys,
            how="left",
            validate="m:1",
        )
