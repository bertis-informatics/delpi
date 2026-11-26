"""Label-free quantification (LFQ) of DelPi search results."""

import logging

import numpy as np
import polars as pl

from delpi.constants import RT_WINDOW_LEN, RT_WINDOW_RADIUS
from delpi.search.result_aggregator import ResultsAggregator
from delpi.search.dia.lfq_utils import perform_lfq

logger = logging.getLogger(__name__)


class LabelFreeQuantifier:
    """Cross-run label-free quantification of DIA or DDA PmSMs.

    Quantifies whatever PmSMs it is given; FDR/q-value filtering is the
    caller's job.
    """

    def __init__(
        self,
        result_aggregator: ResultsAggregator,
        acq_method: str,
        group_key: str = "second_results",
    ):
        self.result_aggregator = result_aggregator
        self.group_key = group_key
        self.acq_method = acq_method.upper()

    def perform_quantification(self, pmsm_df: pl.DataFrame) -> pl.DataFrame:
        """Quantify `pmsm_df`, already restricted to the PmSMs to quantify.

        Returns:
            A DataFrame keyed by ``(run_index, precursor_index)`` with only the
            columns computed here: `ms1_quantity` and, for DIA, `ms2_quantity`,
            `ms2_quantity_normalized` and the normalization QC columns. No input
            column is passed through, so the caller can join them back.
        """
        logger.debug("Starting label-free quantification")

        if self.acq_method == "DIA":
            quant_df = self._quantify_dia(pmsm_df)
            quant_df = self._normalize_ms2_quantity(quant_df)
            # observed_rt is an input column, carried only for the normalization
            quant_df = quant_df.drop("observed_rt")
        elif self.acq_method == "DDA":
            quant_df = self._quantify_dda(pmsm_df)
            # TODO: validate RT-dependent normalization for DDA before enabling
            # quant_df = self._normalize_ms2_quantity(quant_df)
        else:
            raise NotImplementedError()

        return quant_df

    def _quantify_dda(self, pmsm_df: pl.DataFrame) -> pl.DataFrame:
        """Look up each already-assigned PmSM's MS1 area.

        Returns a minimal DataFrame keyed by ``(run_index, precursor_index)``
        with `ms1_quantity`.
        """
        logger.debug("Calculating MS1 areas")
        result_aggregator = self.result_aggregator
        group_key = self.group_key

        dfs = []
        for run_index, sub_df in pmsm_df.group_by("run_index"):
            run_index = run_index[0]
            result_mgr = result_aggregator.get_result_manager(run_index)
            run_ms1_area = result_mgr.read_dict(group_key, data_keys=["ms1_area"])[
                "ms1_area"
            ]
            dfs.append(
                sub_df.select("run_index", "precursor_index").with_columns(
                    pl.Series(
                        "ms1_quantity",
                        run_ms1_area[sub_df["pmsm_index"].to_numpy()],
                        nan_to_null=True,
                    )
                )
            )

        return pl.concat(dfs, how="vertical")

    def _quantify_dia(
        self,
        pmsm_df: pl.DataFrame,
        target_fragments: int = 3,
        min_quant_fragments: int = 3,
        max_fragments: int = 3,
        corr_thresh: float = 0.8,
        min_interference_runs: int = 3,
        interference_min_log2_fold: float = 3.0,
        interference_z_threshold: float = 4.0,
    ) -> pl.DataFrame:
        """Calculate MS1/MS2 peak areas and run cross-run LFQ.

        `pmsm_df` must hold at most one PmSM per ``(run_index, precursor_index)``
        (per-run PmSM assignment guarantees this) and be FDR-filtered already.
        The fragment-selection and interference parameters are passed through to
        :func:`delpi.search.dia.lfq_utils.perform_lfq`.

        Returns a DataFrame keyed by ``(run_index, precursor_index)`` with
        `observed_rt` (for `_normalize_ms2_quantity`), `ms1_quantity` and
        `ms2_quantity`.
        """
        logger.debug("Calculating MS1 and MS2 areas")

        result_aggregator = self.result_aggregator
        sorted_df = pmsm_df.sort("precursor_index", "run_index")

        all_xic_arrays, all_ms1_area_arr = result_aggregator.get_xic_arrays(
            sorted_df, group_key=self.group_key
        )

        idx_df = (
            sorted_df.group_by("precursor_index", maintain_order=True)
            .agg(pl.len())
            .with_columns(pl.col("len").cum_sum().alias("precursor_stop"))
        )
        stop_index_arr = idx_df["precursor_stop"].to_numpy()
        precursor_index_arr = idx_df["precursor_index"].to_numpy()

        all_ms2_ab_arr = perform_lfq(
            precursor_index_arr,
            stop_index_arr,
            all_xic_arrays,
            target_fragments=target_fragments,
            min_quant_fragments=min_quant_fragments,
            max_fragments=max_fragments,
            corr_thresh=corr_thresh,
            min_interference_runs=min_interference_runs,
            interference_min_log2_fold=interference_min_log2_fold,
            interference_z_threshold=interference_z_threshold,
        )

        return sorted_df.select(
            "run_index", "precursor_index", "observed_rt"
        ).with_columns(
            pl.Series("ms1_quantity", all_ms1_area_arr, nan_to_null=True),
            pl.Series("ms2_quantity", all_ms2_ab_arr, nan_to_null=True),
        )

    def _normalize_ms2_quantity(
        self,
        quant_df: pl.DataFrame,
        min_run_fraction: float = 0.5,
        anchor_fraction: float = 0.4,
        min_intensity_quantile: float = 0.1,
        window_size: int = 400,
        min_periods: int = 200,
        rt_column: str = "observed_rt",
    ) -> pl.DataFrame:
        """RT-dependent cross-run normalization of ``ms2_quantity``.

        In the style of DIA-NN/DIA-BERT/PIN/NormalyzerDE RT-local normalization:
        anchors are the precursors whose abundance is most consistent across
        runs after a preliminary global shift; per run, a global log2 shift plus
        a smooth RT-local log2 correction is estimated from the anchors and
        applied to every precursor.

        Adds ``ms2_quantity_normalized`` (same null pattern as ``ms2_quantity``)
        and, unless an early fallback returns, the QC columns
        ``normalization_factor``, ``global_log2_shift`` and ``local_log2_shift``.
        Existing columns and rows are unchanged.
        """
        n_runs = quant_df["run_index"].n_unique()

        # single run: nothing to normalize against
        if n_runs < 2:
            return quant_df.with_columns(
                pl.col("ms2_quantity").alias("ms2_quantity_normalized")
            )

        valid_df = quant_df.filter(
            pl.col("ms2_quantity").is_not_null() & (pl.col("ms2_quantity") > 0)
        ).with_columns(pl.col("ms2_quantity").log(2).alias("log_area"))

        if valid_df.height == 0:
            # no usable quantities
            return quant_df.with_columns(
                pl.col("ms2_quantity").alias("ms2_quantity_normalized")
            )

        # reference_log_area: per-precursor cross-run median log2(ms2_quantity),
        # for precursors observed in >= min_run_fraction of the runs
        min_runs_required = max(2, int(np.ceil(n_runs * min_run_fraction)))
        prec_ref_df = (
            valid_df.group_by("precursor_index")
            .agg(
                pl.col("log_area").median().alias("reference_log_area"),
                pl.col("run_index").n_unique().alias("n_runs_nonnull"),
            )
            .filter(pl.col("n_runs_nonnull") >= min_runs_required)
        )

        if prec_ref_df.height == 0:
            # no reference abundance
            return quant_df.with_columns(
                pl.col("ms2_quantity").alias("ms2_quantity_normalized")
            )

        dev_df = valid_df.join(
            prec_ref_df.select("precursor_index", "reference_log_area"),
            on="precursor_index",
            how="inner",
        ).with_columns(
            (pl.col("log_area") - pl.col("reference_log_area")).alias("deviation")
        )

        # preliminary per-run global shift, only so that globally shifted runs
        # do not bias anchor selection
        prelim_df = dev_df.group_by("run_index").agg(
            pl.col("deviation").median().alias("preliminary_global_shift")
        )
        dev_df = dev_df.join(prelim_df, on="run_index", how="left").with_columns(
            (pl.col("deviation") - pl.col("preliminary_global_shift")).alias("residual")
        )

        # anchors: lowest cross-run MAD of `residual`, excluding the weakest
        # min_intensity_quantile of precursors
        residual_center_df = dev_df.group_by("precursor_index").agg(
            pl.col("residual").median().alias("residual_center")
        )
        mad_df = (
            dev_df.join(residual_center_df, on="precursor_index", how="left")
            .with_columns(
                (pl.col("residual") - pl.col("residual_center")).abs().alias("abs_dev")
            )
            .group_by("precursor_index")
            .agg(pl.col("abs_dev").median().alias("residual_mad"))
        )
        prec_stats_df = prec_ref_df.join(mad_df, on="precursor_index", how="left")

        intensity_cutoff = prec_stats_df["reference_log_area"].quantile(
            min_intensity_quantile
        )
        eligible_df = prec_stats_df.filter(
            pl.col("reference_log_area") >= intensity_cutoff
        )

        if eligible_df.height == 0:
            # nothing survives the weak-precursor filter
            return quant_df.with_columns(
                pl.col("ms2_quantity").alias("ms2_quantity_normalized")
            )

        n_anchors = max(1, int(round(eligible_df.height * anchor_fraction)))
        anchor_precursors = (
            eligible_df.sort("residual_mad").head(n_anchors).select("precursor_index")
        )
        anchor_dev_df = dev_df.join(
            anchor_precursors, on="precursor_index", how="inner"
        )

        # run-global median-ratio normalization over anchors
        global_shift_df = anchor_dev_df.group_by("run_index").agg(
            pl.col("deviation").median().alias("global_log2_shift")
        )

        logger.debug(
            "RT normalization: %d eligible precursors, %d anchors (of %d)",
            eligible_df.height,
            anchor_precursors.height,
            prec_stats_df.height,
        )

        # too few anchors for an RT curve in any run: global shift only
        if anchor_precursors.height < min_periods:
            result_df = quant_df.join(
                global_shift_df, on="run_index", how="left"
            ).with_columns(
                pl.col("global_log2_shift").fill_null(0.0),
                pl.lit(0.0).alias("local_log2_shift"),
            )
        else:
            # RT-local bias: centered rolling median of anchor residuals
            anchor_dev_df = anchor_dev_df.join(
                global_shift_df, on="run_index", how="left"
            ).with_columns(
                (pl.col("deviation") - pl.col("global_log2_shift")).alias(
                    "local_residual"
                )
            )

            local_shift_frames = []
            for run_index in quant_df["run_index"].unique().sort().to_list():
                run_anchor_df = anchor_dev_df.filter(
                    pl.col("run_index") == run_index
                ).sort(rt_column)
                run_quant_df = quant_df.filter(pl.col("run_index") == run_index)
                run_rt_arr = run_quant_df[rt_column].to_numpy()

                if run_anchor_df.height < min_periods:
                    # too few anchors in this run
                    local_shift_arr = np.zeros(run_quant_df.height, dtype=np.float64)
                else:
                    win = min(window_size, run_anchor_df.height)
                    n_min = min(min_periods, run_anchor_df.height)
                    rolling_df = run_anchor_df.select(
                        rt_column,
                        pl.col("local_residual")
                        .rolling_median(window_size=win, min_samples=n_min, center=True)
                        .alias("rolling_shift"),
                    ).drop_nulls("rolling_shift")

                    # collapse duplicate RTs, then linearly interpolate onto
                    # every precursor's RT (edges clamp to nearest anchor value)
                    curve_df = (
                        rolling_df.group_by(rt_column)
                        .agg(pl.col("rolling_shift").median())
                        .sort(rt_column)
                    )
                    if curve_df.height == 0:
                        local_shift_arr = np.zeros(
                            run_quant_df.height, dtype=np.float64
                        )
                    else:
                        curve_rt = curve_df[rt_column].to_numpy()
                        curve_shift = curve_df["rolling_shift"].to_numpy()
                        # center the local curve so it carries no global-level shift
                        curve_shift = curve_shift - np.median(curve_shift)
                        local_shift_arr = np.interp(run_rt_arr, curve_rt, curve_shift)

                logger.debug(
                    "RT normalization run %s: %d anchors, "
                    "local_log2_shift median=%.4f min=%.4f max=%.4f",
                    run_index,
                    run_anchor_df.height,
                    float(np.median(local_shift_arr)),
                    float(np.min(local_shift_arr)),
                    float(np.max(local_shift_arr)),
                )
                local_shift_frames.append(
                    run_quant_df.select("run_index", "precursor_index").with_columns(
                        pl.Series("local_log2_shift", local_shift_arr)
                    )
                )

            local_shift_df = pl.concat(local_shift_frames)
            result_df = (
                quant_df.join(global_shift_df, on="run_index", how="left")
                .join(local_shift_df, on=["run_index", "precursor_index"], how="left")
                .with_columns(
                    pl.col("global_log2_shift").fill_null(0.0),
                    pl.col("local_log2_shift").fill_null(0.0),
                )
            )

        result_df = result_df.with_columns(
            (
                2.0 ** (-(pl.col("global_log2_shift") + pl.col("local_log2_shift")))
            ).alias("normalization_factor")
        ).with_columns(
            pl.when(pl.col("ms2_quantity").is_not_null())
            .then(pl.col("ms2_quantity") * pl.col("normalization_factor"))
            .otherwise(None)
            .alias("ms2_quantity_normalized")
        )

        return result_df
