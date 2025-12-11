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
from delpi.search.dia.lfq_utils import (
    quant_fragment_xics,
    score_fragment_xics,
    find_quantitative_fragments,
)
from delpi.search.dia.peak_token import QUANT_FRAGMENTS

logger = logging.getLogger(__name__)

# Constants
MAX_THEO_INDEX = 15
XIC_LEN = 7


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
        acq_method: str,
        group_key: str = "filtered_results",
        select_topk_fragments: int = 3,
        beta: float = 0.0,
    ):
        self.select_topk_fragments = select_topk_fragments
        self.result_aggregator = result_aggregator
        self.group_key = group_key
        self.acq_method = acq_method.upper()
        self.beta = beta

        unique_precursor_index_arr = None
        for run_index, result_mgr in result_aggregator._results_dict.items():
            data_dict = result_mgr.read_dict(
                group_key,
                data_keys=[
                    "precursor_index",
                ],
            )
            unique_precursor_index_arr = (
                np.union1d(data_dict["precursor_index"], unique_precursor_index_arr)
                if unique_precursor_index_arr is not None
                else data_dict["precursor_index"]
            )

        self.unique_precursor_df = pl.DataFrame(
            {"precursor_index": unique_precursor_index_arr.astype(np.uint32)}
        ).with_row_index("_index")

    def perform_quantification(self) -> pl.DataFrame:
        """
        Perform complete label-free quantification workflow.

        Returns:
            Quantification matrix (n_precursors x n_runs)
        """
        logger.debug("Starting label-free quantification")

        if self.acq_method == "DIA":
            selected_fragments, scores = self._select_quantitative_fragments()
            quant_df = self._calculate_ms2_areas(selected_fragments, scores)
        elif self.acq_method == "DDA":
            quant_df = self._calculate_ms1_areas()
        else:
            raise NotImplementedError()

        return quant_df

    def _select_quantitative_fragments(self) -> None:
        """Score fragment XICs for each run to assess fragment quality."""
        logger.debug("Scoring fragment XICs")

        result_aggregator = self.result_aggregator
        group_key = self.group_key
        topk_fragments = self.select_topk_fragments

        unique_precursor_df = self.unique_precursor_df
        n_precursors = unique_precursor_df.shape[0]
        n_runs = len(result_aggregator._results_dict)

        score_matrix = np.full(
            (n_precursors, n_runs, QUANT_FRAGMENTS), np.nan, dtype=np.float32
        )
        intensity_matrix = np.zeros(
            (n_precursors, n_runs, QUANT_FRAGMENTS), dtype=np.float32
        )

        for run_index, result_mgr in result_aggregator._results_dict.items():
            data_dict = result_mgr.read_dict(
                group_key,
                data_keys=[
                    "precursor_index",
                    "quant_ab",
                    "quant_theo_index",
                    "quant_time_index",
                ],
            )

            frag_score_arr, frag_intensity_arr = score_fragment_xics(
                quant_ab_arr=data_dict["quant_ab"],
                quant_theo_arr=data_dict["quant_theo_index"],
                quant_time_arr=data_dict["quant_time_index"],
            )

            ii = pl.DataFrame({"precursor_index": data_dict["precursor_index"]}).join(
                unique_precursor_df, on="precursor_index", how="left"
            )["_index"]

            score_matrix[ii, run_index, :] = frag_score_arr
            intensity_matrix[ii, run_index, :] = frag_intensity_arr

        topk_index_arr, score_arr = find_quantitative_fragments(
            score_matrix, intensity_matrix, topk=topk_fragments, beta=self.beta
        )

        return topk_index_arr, score_arr

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
                pl.lit(run_index).cast(pl.Int32).alias("run_index")
            )
            dfs.append(quant_df)

        return pl.concat(dfs, how="vertical")

    def _calculate_ms2_areas(
        self, selected_fragments: np.ndarray, scores: np.ndarray
    ) -> pl.DataFrame:
        """Calculate MS2 peak areas using selected fragments."""
        logger.debug("Calculating MS2 areas")

        result_aggregator = self.result_aggregator
        group_key = self.group_key
        unique_precursor_df = self.unique_precursor_df

        dfs = []
        for run_index, result_mgr in result_aggregator._results_dict.items():
            run_name = result_mgr.run_name
            # Load RT mapping
            meta_df = result_mgr.read_df("meta_df").cast(
                {"isolation_win_idx": pl.UInt32}
            )
            quant_dict = result_mgr.read_dict(
                group_key,
                data_keys=[
                    "precursor_index",
                    "ms1_area",
                    "quant_ab",
                    "quant_theo_index",
                    "quant_time_index",
                    "frame_index",
                    "search_group",
                ],
            )

            frame_to_rt_map = self.get_frame_index_to_retention_time_map(meta_df)
            ii = pl.DataFrame({"precursor_index": quant_dict["precursor_index"]}).join(
                unique_precursor_df, on="precursor_index", how="left"
            )["_index"]

            areas = quant_fragment_xics(
                frame_to_rt_map_arr=frame_to_rt_map,
                quant_ab_arr=quant_dict["quant_ab"],
                quant_theo_arr=quant_dict["quant_theo_index"],
                quant_time_arr=quant_dict["quant_time_index"],
                selected_index_arr=selected_fragments[ii, :],
                frag_score_arr=scores[ii, :],
                frame_index_arr=quant_dict["frame_index"],
                win_index_arr=quant_dict["search_group"],
            )
            quant_df = pl.DataFrame(
                {
                    "precursor_index": quant_dict["precursor_index"],
                    "ms1_area": quant_dict["ms1_area"],
                    "ms2_area": areas,
                }
            ).with_columns(pl.lit(run_index).cast(pl.Int32).alias("run_index"))

            dfs.append(quant_df)
            # result_mgr.write_dict(group_key, {"ms2_area": areas})
            logger.debug(f"Calculated MS2 areas for {run_name}")

        return pl.concat(dfs, how="vertical")

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
