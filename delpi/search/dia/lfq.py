"""
Label-Free Quantification (LFQ) Module for DelPi DIA Search

This module provides functionality for performing label-free quantification across
multiple DIA runs, including:
- Fragment scoring and selection
- MS1 and MS2 area calculations
- Cross-run quantification matrix generation
- Integration with ResultManager for data handling
"""

import logging

import numpy as np
import polars as pl

from delpi.search.result_aggregator import ResultsAggregator
from delpi.search.dia.pmsm_lfq import perform_lfq

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
        pmsm_df: pl.DataFrame,
        result_aggregator: ResultsAggregator,
        q_value_cutoff: float,
        acq_method: str,
        group_key: str = "second_results",
    ):
        self.pmsm_df = pmsm_df
        self.result_aggregator = result_aggregator
        self.q_value_cutoff = q_value_cutoff
        self.group_key = group_key
        self.acq_method = acq_method.upper()

    def perform_quantification(self) -> pl.DataFrame:
        """
        Perform complete label-free quantification workflow.

        Returns:
            Quantification matrix (n_precursors x n_runs)
        """
        logger.debug("Starting label-free quantification")

        if self.acq_method == "DIA":
            quant_df = self._calculate_ms2_areas()
        elif self.acq_method == "DDA":
            quant_df = self._calculate_ms1_areas()
        else:
            raise NotImplementedError()

        return quant_df

    def _calculate_ms1_areas(self) -> pl.DataFrame:
        """Calculate MS2 peak areas using selected fragments."""
        logger.debug("Calculating MS2 areas")
        result_aggregator = self.result_aggregator
        group_key = self.group_key

        dfs = []
        for run_index, result_mgr in result_aggregator._results_dict.items():
            quant_dict = result_mgr.read_dict(
                group_key,
                data_keys=["precursor_index", "ms1_area"],
            )
            quant_df = pl.DataFrame(quant_dict).with_columns(
                pl.lit(run_index).cast(pl.UInt32).alias("run_index")
            )
            dfs.append(quant_df)

        return pl.concat(dfs, how="vertical")

    def _calculate_ms2_areas(self) -> pl.DataFrame:
        """Calculate MS2 peak areas using selected fragments."""
        logger.debug("Calculating MS2 areas")

        result_aggregator = self.result_aggregator
        pmsm_df = self.pmsm_df
        q_value_cutoff = self.q_value_cutoff

        target_pmsm_df = pmsm_df.filter(
            (pl.col("is_decoy") == False)
            & (
                pl.col("global_precursor_q_value").min().over("precursor_index")
                <= q_value_cutoff
            )
        ).sort(pl.col("precursor_index", "run_index", "pmsm_index"))

        all_xic_arrays, all_ms1_area_arr = result_aggregator.get_xic_arrays(
            target_pmsm_df, group_key=self.group_key
        )

        ## find reference run for each precursor
        ref_run_df = (
            target_pmsm_df.group_by(
                ["precursor_index", "run_index"], maintain_order=True
            )
            .agg(pl.len(), pl.col("logit").sum())
            .group_by("precursor_index", maintain_order=True)
            .agg(
                pl.col("run_index").sort_by(["len", "logit"]).last(),
                pl.col("run_index").n_unique().alias("run_count"),
                pl.col("len").sum().alias("pmsm_count"),
            )
            .with_columns(
                pl.col("pmsm_count").cum_sum().alias("pmsm_stop"),
                pl.col("run_count").cum_sum(),
            )
        )

        all_run_index_arr = target_pmsm_df["run_index"].to_numpy()
        all_rt_arr = target_pmsm_df["observed_rt"].to_numpy()

        precursor_indices = ref_run_df["precursor_index"].to_numpy()
        ref_run_indices = ref_run_df["run_index"].to_numpy()
        run_stop_indices = ref_run_df["run_count"].to_numpy()
        pmsm_stop_indices = ref_run_df["pmsm_stop"].to_numpy()

        (
            quant_precursor_index_arr,
            quant_run_index_arr,
            quant_rt_arr,
            quant_ab_arr,
            quant_ms1_ab_arr,
        ) = perform_lfq(
            num_runs=len(result_aggregator._results_dict),
            all_run_index_arr=all_run_index_arr,
            all_rt_arr=all_rt_arr,
            all_xic_arrays=all_xic_arrays,
            all_ms1_area_arr=all_ms1_area_arr,
            precursor_indices=precursor_indices,
            ref_run_indices=ref_run_indices,
            run_stop_indices=run_stop_indices,
            pmsm_stop_indices=pmsm_stop_indices,
        )

        quant_df = pl.DataFrame(
            {
                "run_index": quant_run_index_arr,
                "precursor_index": quant_precursor_index_arr,
                "quantification_rt": quant_rt_arr,
                "ms1_area": quant_ms1_ab_arr,
                "ms2_area": quant_ab_arr,
            },
            nan_to_null=True,
        )
        return quant_df

    @staticmethod
    def get_frame_index_to_retention_time_map(meta_df) -> np.ndarray:
        """Generate RT mapping from metadata.
        returns rt_array of [#windows, #frames]
        """
        max_win_idx, max_frame_count = (
            meta_df.filter(pl.col("isolation_win_idx").is_not_null())
            .group_by("isolation_win_idx")
            .agg(pl.len())
            .select(pl.col("isolation_win_idx").max(), pl.col("len").max())
        ).row(0)

        rt_map_arr = np.empty((max_win_idx + 1, max_frame_count), dtype=np.float32)

        for win_idx_, sub_df in meta_df.filter(
            pl.col("isolation_win_idx").is_not_null()
        ).group_by("isolation_win_idx"):
            rt_arr = sub_df["time_in_seconds"].to_numpy()
            rt_map_arr[win_idx_[0], : rt_arr.shape[0]] = rt_arr

        return rt_map_arr
