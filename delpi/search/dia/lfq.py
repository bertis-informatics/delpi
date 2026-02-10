"""
Label-Free Quantification (LFQ) Module for DelPi DIA Search
"""

import logging

import numpy as np
import polars as pl

from delpi.search.result_aggregator import ResultsAggregator
from delpi.search.dia.lfq_utils import perform_lfq

logger = logging.getLogger(__name__)


class LabelFreeQuantifier:
    """
    Handles label-free quantification across multiple DIA search results.

    This class coordinates quantification workflow including:
    - Loading search results from multiple HDF5 files
    - Fragment scoring and selection
    - MS1/MS2 area calculations
    - Cross-run quantification matrix generation
    """

    def __init__(
        self,
        result_aggregator: ResultsAggregator,
        q_value_cutoff: float,
        acq_method: str,
        group_key: str = "second_results",
    ):
        self.result_aggregator = result_aggregator
        self.q_value_cutoff = q_value_cutoff
        self.group_key = group_key
        self.acq_method = acq_method.upper()

    def perform_quantification(self, pmsm_df: pl.DataFrame) -> pl.DataFrame:
        """
        Perform complete label-free quantification workflow.

        Returns:
            Quantification matrix (n_precursors x n_runs)
        """
        logger.debug("Starting label-free quantification")

        if self.acq_method == "DIA":
            quant_df = self._quantify_dia(pmsm_df)
            # quant_df = self._normalize_ms2_area(quant_df)
        elif self.acq_method == "DDA":
            quant_df = self._quantify_dda()
        else:
            raise NotImplementedError()

        return quant_df

    def _quantify_dda(self) -> pl.DataFrame:

        logger.debug("Calculating MS1 areas")
        result_aggregator = self.result_aggregator
        group_key = self.group_key

        dfs = []
        for run_index, result_mgr in result_aggregator._results_dict.items():
            quant_dict = result_mgr.read_dict(
                group_key,
                data_keys=["precursor_index", "ms1_area"],
            )
            quant_df = pl.DataFrame(quant_dict, nan_to_null=True).with_columns(
                pl.lit(run_index).cast(pl.UInt32).alias("run_index")
            )
            dfs.append(quant_df)

        return pl.concat(dfs, how="vertical")

    def _quantify_dia(self, pmsm_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate MS2 peak areas using selected fragments."""
        logger.debug("Calculating MS1 and MS2 areas")

        result_aggregator = self.result_aggregator
        q_value_cutoff = self.q_value_cutoff

        target_pmsm_df = (
            pmsm_df.filter(
                (pl.col("is_decoy") == False)
                & (pl.col("global_precursor_q_value") <= q_value_cutoff)
                & (pl.col("precursor_q_value") <= q_value_cutoff)
            )
            .with_columns(
                pl.col("score")
                .max()
                .over(["precursor_index", "run_index"])
                .alias("max_precursor_score"),
            )
            .filter(
                # filter out low confidence PmSMs
                (pl.col("score") / pl.col("max_precursor_score") > 0.5)
                | (pl.col("max_precursor_score") - pl.col("score") < 1.0)
            )
            .sort(pl.col("precursor_index", "run_index"))
        )

        all_xic_arrays, all_ms1_area_arr = result_aggregator.get_xic_arrays(
            target_pmsm_df, group_key=self.group_key
        )

        ## compute median intensity at apex time point
        med_intensity = np.median(
            all_xic_arrays[:, :, all_xic_arrays.shape[-1] // 2], axis=1
        )
        target_pmsm_df = target_pmsm_df.with_columns(
            pl.Series(name="med_intensity", values=med_intensity)
        ).with_row_index("index_")

        ## select a PmSM for each precursor based on the median intensity
        selected_pmsm_df = target_pmsm_df.group_by(
            ["precursor_index", "run_index"], maintain_order=True
        ).agg(
            pl.all().sort_by("med_intensity").last(),
        )

        ## estimate RT deviation median for each precursor
        rt_diff_df = (
            selected_pmsm_df.select(
                pl.col("precursor_index"),
                (pl.col("observed_rt") - pl.col("predicted_rt")).abs().alias("rt_diff"),
            )
            .group_by("precursor_index")
            .agg(pl.col("rt_diff").median().alias("median_rt_diff"))
        )

        ## re-select PmSMs considering RT deviation
        selected_pmsm_df = (
            target_pmsm_df.join(
                rt_diff_df,
                on="precursor_index",
                how="left",
            )
            .with_columns(
                rt_match=(pl.col("observed_rt") - pl.col("predicted_rt")).abs()
                < pl.col("median_rt_diff") * 3
            )
            .group_by(["precursor_index", "run_index"], maintain_order=True)
            .agg(
                pl.all().sort_by("rt_match", "med_intensity").last(),
            )
        )

        ## perform LFQ using selected PmSMs
        idx_df = (
            selected_pmsm_df.group_by(["precursor_index"], maintain_order=True)
            .agg(
                pl.len(),
            )
            .with_columns(pl.col("len").cum_sum().alias("precursor_stop"))
        )
        stop_index_arr = idx_df["precursor_stop"].to_numpy()
        precursor_index_arr = idx_df["precursor_index"].to_numpy()
        all_xic_arr = all_xic_arrays[selected_pmsm_df["index_"]]
        all_ms1_ab_arr = all_ms1_area_arr[selected_pmsm_df["index_"]]
        all_ms2_ab_arr = perform_lfq(
            precursor_index_arr,
            stop_index_arr,
            all_xic_arr,
            min_fragments=6,
            max_fragments=9,
            corr_thresh=0.9,
        )

        quant_df = selected_pmsm_df.select(
            pl.col("run_index", "precursor_index"),
            pl.col("observed_rt").alias("quantification_rt"),
            pl.Series(name="ms1_area", values=all_ms1_ab_arr, nan_to_null=True),
            pl.Series(name="ms2_area", values=all_ms2_ab_arr, nan_to_null=True),
        )

        return quant_df

    def _normalize_ms2_area(
        self,
        quant_df: pl.DataFrame,
        p: float = 0.25,  # housekeeping fraction (0~1)
        # use_median_ref: bool = True,  # reference = median(sum_hk) else mean
        eps: float = 1e-9,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Return:
        - quant_df with new columns: ms2_area_norm, run_scaling_factor
        - hk_summary: precursor CV table (for debugging / inspection)

        Housekeeping selection:
        - compute per-precursor CV across runs using area_col
        - keep precursors with enough non-null runs
        - pick lowest-CV top p fraction
        Scaling:
        - per run: sum areas over housekeeping precursors -> sum_hk[run]
        - scaling_factor[run] = sum_hk[run] / reference(sum_hk)
        - normalized_area = area / scaling_factor
        """

        num_runs = len(self.result_aggregator._results_dict)

        if num_runs < 2:
            return quant_df.with_columns(
                pl.col("ms2_area").alias("normalized_ms2_area")
            )

        min_nonnull_runs = max(2, int(round(num_runs * 0.5)))

        # 1) precursor-level stats across runs (nonnull only)
        prec_stats = (
            quant_df.filter(pl.col("ms2_area").is_not_null() & (pl.col("ms2_area") > 0))
            .group_by("precursor_index")
            .agg(
                [
                    pl.col("run_index").n_unique().alias("n_runs_nonnull"),
                    pl.col("ms2_area").mean().alias("mean_area"),
                    pl.col("ms2_area").std(ddof=1).alias("std_area"),
                ]
            )
            .filter(pl.col("n_runs_nonnull") > 1)
            .with_columns(
                (pl.col("std_area") / (pl.col("mean_area") + eps)).alias("cv"),
            )
            .filter(pl.col("n_runs_nonnull") >= min_nonnull_runs)
            .sort("cv")
        )

        if prec_stats.height == 0:
            # raise ValueError(
            #     "No precursors eligible for housekeeping selection. "
            #     "Try lowering min_nonnull_runs or check ms2_area null/zero distribution."
            # )
            return quant_df

        # 2) pick housekeeping precursors = lowest CV top p fraction
        k = max(1, int(round(prec_stats.height * p)))
        hk_precursors = prec_stats.head(k).select("precursor_index")

        # 3) compute run-wise sum over housekeeping precursors
        run_scale_df = (
            quant_df.join(hk_precursors, on="precursor_index", how="inner")
            .filter(pl.col("ms2_area").is_not_null() & (pl.col("ms2_area") > 0))
            .group_by("run_index")
            .agg(pl.col("ms2_area").sum().alias("sum_hk"))
            .sort("run_index")
        )

        if run_scale_df.height == 0:
            run_scale_df = (
                quant_df.filter(
                    pl.col("ms2_area").is_not_null() & (pl.col("ms2_area") > 0)
                )
                .group_by("run_index")
                .agg(pl.col("ms2_area").sum().alias("sum"))
                .sort("run_index")
            )

        # scaling_factor
        run_scale_df = run_scale_df.with_columns(
            (pl.col("sum_hk") / (pl.col("sum_hk").median() + eps)).alias(
                "run_scaling_factor"
            )
        )

        # 4) apply normalization
        quant_df = (
            quant_df.join(
                run_scale_df.select(["run_index", "run_scaling_factor"]),
                on="run_index",
                how="left",
            )
            .with_columns(
                (pl.col("ms2_area") / (pl.col("run_scaling_factor") + eps)).alias(
                    f"normalized_ms2_area"
                )
            )
            .drop("run_scaling_factor")
        )

        return quant_df
