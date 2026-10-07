"""Global and run-specific FDR estimation from precomputed scores."""

import polars as pl

from delpi.utils.fdr import calculate_q_value


class FDRAnalyzer:
    """Calculate q-values without performing protein inference or scoring.

    The caller supplies protein-group membership and scores, when protein
    FDR is wanted. PmSM and protein-group score columns are independently
    selectable on each analysis call; higher scores must mean better hits.
    No sequence database or inference configuration is needed here.
    """

    def perform_global_analysis(
        self,
        pmsm_df: pl.DataFrame,
        score_column: str = "score",
        protein_group_score_column: str = "protein_group_score",
        target_to_decoy_size_ratio: float = 1.0,
    ) -> pl.DataFrame:
        """Calculate global precursor, peptide and optional protein-group FDR.

        Precursor/peptide confidence uses the maximum supplied PmSM score
        across all runs. Protein-group confidence uses the supplied group
        score, repeated on its member PmSMs. For independently calculated
        global group scores, specify e.g. ``global_protein_group_score``.
        Existing q-value columns are replaced when analysis is repeated.
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
        """Calculate q-values per group and attach them to the original rows.

        Each group contributes its highest finite ``score_column`` value
        and the corresponding ``is_decoy`` label. Rows with null keys or
        scores are excluded from estimation. ``out_column`` is replaced if
        already present; groups without valid evidence receive null q-values.
        The supplied frame defines the analysis scope (all runs or one run).
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
            # Preserve the key order of the protein-scoring output when
            # collapsing its scores repeated across member PmSM rows.
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
