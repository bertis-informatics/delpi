"""
DIA Search Engine for DelPi

This module provides the DIA-specific search engine implementation.
"""

from collections import defaultdict
from typing import Dict
import logging
import time

import polars as pl
import numpy as np
import torch
from torch.amp import autocast
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from delpi.lcms.reader.base import MassSpecData
from delpi.lcms.dia_run import DIARun, DIAWindow
from delpi.database.spec_lib_reader import SpectralLibReader
from delpi.database.numba.spec_lib_container import SpectralLibContainer
from delpi.search.config import SearchConfig
from delpi.search.base_engine import BaseSearchEngine
from delpi.model.delpi_model import DelPiModel
from delpi.lcms.data_container import PeakContainer
from delpi.search.result_manager import ResultManager
from delpi.model.rt_calibrator import RetentionTimeCalibrator
from delpi.search.search_state import SearchState
from delpi.search.dia.quick_search import run_quick_search
from delpi.search.dia.peak_group import find_peak_groups
from delpi.search.dia.batch_generator import count_total_batches, generate_batches
from delpi.search.dia.lfq_utils import get_ms1_area
from delpi.search.clustering import cluster_matches
from delpi.model.input import THEORETICAL_PEAK, EXPERIMENTAL_PEAK
from delpi.search.dia.peak_token import QUANT_THEO_IDX, QUANT_AB_IDX, QUANT_TIME_IDX


logger = logging.getLogger(__name__)

LOGIT_CUTOFF = 1.0
TOPK_PER_PRECURSOR = 10


class DIASearchEngine(BaseSearchEngine):
    """
    Data-Independent Acquisition (DIA) search engine implementation.

    This engine implements the DIA-specific search pipeline:
    1. DIA isolation scheme detection and window analysis
    2. Quick search for initial retention time calibration
    3. Full DIA search with deep learning scoring across windows
    4. Multi-stage retention time calibration refinement
    5. Quantitative fragment selection for label-free quantification

    The engine processes DIA data by first understanding the isolation
    scheme, then performing a rapid search to establish RT relationships,
    followed by comprehensive scoring across all windows.
    """

    def get_acquisition_method(self) -> str:
        """Return the acquisition method for DIA."""
        return "DIA"

    def _search_spectra(
        self,
        search_config: SearchConfig,
        dia_win: DIAWindow,
        speclib_container: SpectralLibContainer,
        ms1_peak_df: PeakContainer,
        model: DelPiModel,
        X_theo_tensor: torch.Tensor,
        X_exp_tensor: torch.Tensor,
        batch_size: int = 512,
        peak_group_topk: int = 10,
        logit_cutoff: float = LOGIT_CUTOFF,
    ) -> Dict[str, np.ndarray]:
        """Search DIA spectra for a specific isolation window."""
        ms1_tol = search_config["ms1_mass_tol_in_ppm"]
        ms2_tol = search_config["ms2_mass_tol_in_ppm"]

        frame_num_map = dia_win.get_frame_num_map()
        ms2_peak_df = dia_win.get_peak_container()
        ms1_rt_arr = dia_win.ms1_meta_df["time_in_seconds"].to_numpy()

        peak_group_container, peak_index_container = find_peak_groups(
            speclib_container,
            ms1_peak_df=ms1_peak_df,
            ms2_peak_df=ms2_peak_df,
            frame_num_map=frame_num_map,
            ms1_mass_tol=ms1_tol,
            ms2_mass_tol=ms2_tol,
            topk=peak_group_topk,
        )
        total_batches = count_total_batches(
            peak_group_container.peak_count_arr, batch_size=batch_size
        )
        batch_iter = generate_batches(
            speclib_container,
            ms1_peak_df,
            ms2_peak_df,
            peak_group_container,
            peak_index_container,
            frame_num_map,
            batch_size=batch_size,
            ms1_mass_tol=ms1_tol,
            ms2_mass_tol=ms2_tol,
        )

        results = defaultdict(list)
        for (
            precursor_index_arr,
            frame_num_arr,
            x_theo,
            x_exp,
            x_ind,
            x_quant,
            ms1_scale_arr,
        ) in tqdm(
            batch_iter, total=total_batches, position=1, desc="PMSMs", leave=False
        ):
            n = x_theo.shape[0]
            X_theo = X_theo_tensor[:n].copy_(torch.from_numpy(x_theo))
            X_exp = X_exp_tensor[:n, : x_exp.shape[1], :].copy_(torch.from_numpy(x_exp))

            logits, x_feature = model(X_theo, X_exp)

            logits = logits.flatten()
            mask = logits > logit_cutoff

            x_feature = x_feature[mask].detach().cpu().numpy()
            logits = logits[mask].detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            precursor_index_arr = precursor_index_arr[mask]
            frame_num_arr = frame_num_arr[mask]
            frame_index_arr = dia_win.frame_num_to_index[frame_num_arr]
            x_ind = x_ind[mask]
            x_quant = x_quant[mask]
            ms1_area_arr = get_ms1_area(
                ms1_rt_arr, frame_index_arr, x_exp[mask], ms1_scale_arr[mask]
            )

            observed_rt = frame_num_map.ms2_rt_arr[
                frame_num_map.frame_num_to_index_arr[frame_num_arr]
            ]

            if precursor_index_arr.shape[0] > 0:
                results["precursor_index"].append(precursor_index_arr)
                results["frame_num"].append(frame_num_arr)
                results["frame_index"].append(frame_index_arr)
                results["logit"].append(logits)
                results["features"].append(x_feature)
                results["observed_rt"].append(observed_rt)
                results["peak_indices"].append(x_ind)
                results["ms1_area"].append(ms1_area_arr)
                results["quant_ab"].append(
                    x_quant[..., QUANT_AB_IDX].astype(np.float32)
                )
                results["quant_theo_index"].append(
                    x_quant[..., QUANT_THEO_IDX].astype(np.int8)
                )
                results["quant_time_index"].append(
                    x_quant[..., QUANT_TIME_IDX].astype(np.int8)
                )

        results = {k: np.concatenate(v) for k, v in results.items()}
        if len(results) > 0:
            results["search_group"] = np.full(
                results["precursor_index"].shape[0],
                fill_value=dia_win.window_index,
                dtype=np.uint32,
            )

        return results

    def _perform_full_search(
        self,
        run: DIARun,
        rt_calibrator: RetentionTimeCalibrator,
        batch_size: int = 512,
        logit_cutoff: float = LOGIT_CUTOFF,
    ) -> ResultManager:
        """Perform the complete DIA search workflow."""

        search_config = self.search_config
        result_manager = search_config.get_result_manager(run.name)
        model = self._load_model()
        group_key = self.get_results_group_key()

        speclib_reader = SpectralLibReader(peptide_db_path=self.get_db_dir())
        speclib_reader.calibrate_rt(rt_calibrator)

        ms1_map = run.get_ms1_map()
        ms1_peak_df = ms1_map.get_peak_container()
        X_theo_tensor = torch.empty(
            (batch_size, 19, len(THEORETICAL_PEAK)),
            dtype=torch.float16,
            device=self.device,
        )
        X_exp_tensor = torch.rand(
            (batch_size, 512, len(EXPERIMENTAL_PEAK)),
            dtype=torch.float16,
            device=self.device,
        )

        with torch.inference_mode(), autocast(
            dtype=torch.float16, device_type=self.device.type
        ), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            cluster_count = 0
            for win_idx, dia_win in tqdm(
                run.windows.items(), position=0, desc="Isolation-Window", leave=True
            ):
                speclib_container = speclib_reader.read_by_mz_range(
                    *dia_win.isolation_mz_range
                )

                # no precursor candidates in this window
                if speclib_container is None:
                    continue

                results = self._search_spectra(
                    search_config,
                    dia_win,
                    speclib_container,
                    ms1_peak_df,
                    model,
                    X_theo_tensor,
                    X_exp_tensor,
                    batch_size,
                    peak_group_topk=TOPK_PER_PRECURSOR,
                    logit_cutoff=logit_cutoff,
                )

                # Cluster matches sharing peaks
                if (
                    results.get("frame_num") is not None
                    and results["frame_num"].shape[0] > 0
                ):
                    cluster_arr = cluster_matches(
                        frame_index_arr=dia_win.frame_num_to_index[
                            results["frame_num"]
                        ],
                        peak_index_arr=results["peak_indices"],
                        max_frame_diff=1,
                        jaccard_thres=0.6,
                    )
                    results["cluster"] = cluster_arr + cluster_count
                    cluster_count += 1 + np.max(cluster_arr)

                result_manager.write_dict(group_key, results)

        self.empty_device_cache()

        return result_manager

    def perform_search(self, lcms_data: MassSpecData) -> ResultManager:
        """
        Perform the DIA search pipeline.

        Args:
            search_config: Search configuration
            lcms_data: DIA LC-MS/MS data

        Returns:
            ResultManager with search results
        """
        # Prepare DIA data and return DIARun object
        run = DIARun(lcms_data)
        num_wins = run.dia_scheme_df.shape[0]

        iso_min_mz = run.dia_scheme_df.item(0, "isolation_min_mz")
        iso_max_mz = run.dia_scheme_df.item(-1, "isolation_max_mz")
        logger.info(
            f"Detected DIA scheme: {num_wins} isolation windows for {iso_min_mz:.2f} - {iso_max_mz:.2f} m/z"
        )

        run.load_windows(free_ms_data=False)
        logger.info("DIA data prepared")

        # RT calibration with quick search or previous search results
        rt_calibrator = self._perform_rt_calibration(run, before_full_search=True)

        # Full search
        st_t = time.perf_counter()
        logger.info("Search started")
        logit_cutoff = (
            LOGIT_CUTOFF
            if self.state < SearchState.SECOND_SEARCH
            else (LOGIT_CUTOFF - 2.0)
        )
        result_manager = self._perform_full_search(
            run, rt_calibrator, batch_size=512, logit_cutoff=logit_cutoff
        )
        elapsed = time.perf_counter() - st_t
        logger.info(f"Search completed. Elapsed: {elapsed:.1f} s")

        # Re-fit RT calibrator with full search results and save predicted RTs for re-scoring
        rt_calibrator = self._perform_rt_calibration(run, before_full_search=False)

        if self.state == SearchState.FIRST_SEARCH:
            result_manager.write_df(
                df=run.meta_df.select(pl.exclude("peak_start", "peak_stop")),
                key="meta_df",
            )
            result_manager.write_attr(
                "cycle_time_in_seconds", run.cycle_time_in_seconds
            )
            result_manager.write_attr(
                "gradient_length_in_seconds", run.gradient_length_in_seconds
            )
            result_manager.write_attr("xic_peak_interval", run.cycle_time_in_seconds)

        return result_manager

    def perform_quick_search(self, run: DIARun) -> pl.DataFrame:
        pmsm_df = run_quick_search(
            run,
            db_dir=self.search_config.db_dir,
            ms2_tol_in_ppm=self.search_config["ms2_mass_tol_in_ppm"],
            min_matches=500000,
            q_value_cutoff=0.05,
        )

        return pmsm_df
