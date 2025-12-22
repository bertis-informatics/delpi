"""
DDA Search Engine for DelPi

This module provides the DDA-specific search engine implementation.
"""

import time
import logging
from collections import defaultdict
from typing import Dict

import torch
from torch.amp import autocast
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
import numpy as np
import polars as pl

from delpi.lcms.reader.base import MassSpecData
from delpi.model.classifier import DelPiModel
from delpi.lcms.dda_run import DDARun
from delpi.lcms.ms2_spectra import MS2Spectra
from delpi.lcms.data_container import PeakContainer, MetaContainer
from delpi.database.spec_lib_reader import SpectralLibReader
from delpi.database.numba.spec_lib_container import SpectralLibContainer
from delpi.search.config import SearchConfig
from delpi.search.result_manager import ResultManager
from delpi.search.search_state import SearchState
from delpi.search.base_engine import BaseSearchEngine
from delpi.search.dda.peak_group import find_peak_groups
from delpi.search.dda.batch_generator import count_total_batches, generate_batches
from delpi.search.clustering import cluster_matches
from delpi.search.dia.lfq_utils import get_ms1_area_dda
from delpi.constants import ISOLATION_LOWER_TOL, ISOLATION_UPPER_TOL
from delpi.model.input import THEORETICAL_PEAK, EXPERIMENTAL_PEAK


logger = logging.getLogger(__name__)

LOGIT_CUTOFF = 1.0
TOPK_PER_PRECURSOR = 10


class DDASearchEngine(BaseSearchEngine):
    """
    Data-Dependent Acquisition (DDA) search engine implementation.

    This engine implements the DDA-specific search pipeline:
    1. MS2 peak grouping and precursor isolation
    2. Fragment database matching with mass tolerance
    3. Deep learning-based peptide-spectrum match scoring
    4. Retention time calibration and prediction
    5. Match clustering to resolve shared peaks

    The engine processes DDA data by isolation windows, grouping peaks
    into precursor-fragment relationships and scoring them using a
    trained neural network model.
    """

    def __init__(
        self,
        search_config: SearchConfig,
        device: torch.device,
        state: SearchState = SearchState.INIT,
    ):
        super().__init__(search_config, device, state)
        self.lc_peak_width = 10

    def get_acquisition_method(self) -> str:
        """Return the acquisition method for DDA."""
        return "DDA"

    def _search_spectra(
        self,
        ms2_map: MS2Spectra,
        speclib_container: SpectralLibContainer,
        ms1_meta_df: MetaContainer,
        ms1_peak_df: PeakContainer,
        model: DelPiModel,
        X_theo_tensor: torch.Tensor,
        X_exp_tensor: torch.Tensor,
        rt_window_half: float,
        batch_size: int = 512,
        peak_group_topk: int = 10,
        logit_cutoff: float = LOGIT_CUTOFF,
    ) -> Dict[str, np.ndarray]:
        """Search DDA spectra for a specific isolation window."""

        search_config = self.search_config
        ms1_tol = search_config["ms1_mass_tol_in_ppm"]
        ms2_tol = search_config["ms2_mass_tol_in_ppm"]

        ms2_meta_df = ms2_map.get_meta_container()
        ms2_peak_df = ms2_map.get_peak_container()

        peak_group_container, peak_index_container = find_peak_groups(
            speclib_container,
            ms1_meta_df,
            ms2_meta_df,
            ms1_peak_df,
            ms2_peak_df,
            rt_window_half,
            isolation_lower_tol_in_da=ISOLATION_LOWER_TOL,
            isolation_upper_tol_in_da=ISOLATION_UPPER_TOL,
            topk=peak_group_topk,
        )

        total_batches = count_total_batches(
            peak_group_container.peak_count_arr, batch_size=512
        )

        batch_iter = generate_batches(
            speclib_container,
            ms1_meta_df,
            ms2_meta_df,
            ms1_peak_df,
            ms2_peak_df,
            peak_group_container,
            peak_index_container,
            ms1_mass_tol=ms1_tol,
            ms2_mass_tol=ms2_tol,
            batch_size=batch_size,
        )

        results = defaultdict(list)
        for (
            precursor_index_arr,
            frame_num_arr,
            x_theo,
            x_exp,
            x_ind,
            ms1_scale_arr,
        ) in tqdm(
            batch_iter,
            total=total_batches,
            position=1,
            desc="PMSMs",
            leave=False,
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
            x_ind = x_ind[mask]

            ms1_area_arr = get_ms1_area_dda(x_exp[mask], ms1_scale_arr[mask])

            observed_rt = ms2_meta_df.rt_arr[
                ms2_meta_df.frame_num_to_index_arr[frame_num_arr]
            ]

            if precursor_index_arr.shape[0] > 0:
                results["precursor_index"].append(precursor_index_arr)
                results["frame_num"].append(frame_num_arr)
                results["logit"].append(logits)
                results["features"].append(x_feature)
                results["peak_indices"].append(x_ind)
                results["observed_rt"].append(observed_rt)
                results["ms1_area"].append(ms1_area_arr)

        results = {k: np.concatenate(v) for k, v in results.items()}
        if len(results) > 0:
            results["search_group"] = np.full(
                results["precursor_index"].shape[0],
                fill_value=ms2_map.group_index,
                dtype=np.uint32,
            )

        return results

    def _perform_full_search(
        self,
        run: DDARun,
        batch_size: int = 512,
        logit_cutoff: float = LOGIT_CUTOFF,
    ) -> ResultManager:
        """Perform the complete DDA search workflow."""

        search_config = self.search_config
        result_manager = search_config.get_result_manager(run.name)
        model = self._load_model()
        group_key = self.get_results_group_key()

        speclib_reader = SpectralLibReader(peptide_db_path=self.get_db_dir())

        if self.state > SearchState.FIRST_SEARCH:
            lc_peak_width = result_manager.read_attr("lc_peak_width")
        else:
            lc_peak_width = run.estimate_LC_fwhm() * 1.69

        rt_window_half = lc_peak_width * 0.5  # in terms of seconds
        self.lc_peak_width = lc_peak_width

        ms1_meta_df = run.ms1_map.get_meta_container()
        ms1_peak_df = run.ms1_map.get_peak_container()

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
            for _, ms2_map in tqdm(
                run.ms2_maps.items(), position=0, desc="Isolation-Window", leave=True
            ):
                speclib_container = speclib_reader.read_by_isolation_window(
                    min_isolation_mz=ms2_map.isolation_mz_range[0],
                    max_isolation_mz=ms2_map.isolation_mz_range[1],
                    isolation_lower_tol_in_da=ISOLATION_LOWER_TOL,
                    isolation_upper_tol_in_da=ISOLATION_UPPER_TOL,
                )

                # no precursor candidates in this window
                if speclib_container is None:
                    continue

                results = self._search_spectra(
                    ms2_map,
                    speclib_container,
                    ms1_meta_df,
                    ms1_peak_df,
                    model,
                    X_theo_tensor,
                    X_exp_tensor,
                    rt_window_half,
                    batch_size=batch_size,
                    peak_group_topk=TOPK_PER_PRECURSOR,
                    logit_cutoff=logit_cutoff,
                )

                # Cluster matches sharing peaks
                if (
                    results.get("frame_num") is not None
                    and results["frame_num"].shape[0] > 0
                ):
                    cluster_arr = cluster_matches(
                        frame_index_arr=results["frame_num"],
                        peak_index_arr=results["peak_indices"],
                        max_frame_diff=0,
                        jaccard_thres=0.6,
                    )
                    results["cluster"] = cluster_arr + cluster_count
                    cluster_count += 1 + np.max(cluster_arr)

                result_manager.write_dict(group_key, results)

        self.empty_device_cache()

        return result_manager

    def perform_quick_search(self, run: DDARun) -> pl.DataFrame:
        # In DDA search, RT search space is not reduced
        return None

    def perform_search(self, lcms_data: MassSpecData) -> ResultManager:
        """
        Perform the DDA search pipeline.

        Args:
            search_config: Search configuration
            lcms_data: DDA LC-MS/MS data

        Returns:
            ResultManager with search results
        """
        run = DDARun(lcms_data)
        run = run.prepare(num_groups=128)
        logger.info("DDA data prepared")

        st_t = time.perf_counter()
        logger.info("Search started")
        logit_cutoff = (
            LOGIT_CUTOFF
            if self.state < SearchState.SECOND_SEARCH
            else (LOGIT_CUTOFF - 2.0)
        )
        result_manager = self._perform_full_search(
            run, batch_size=512, logit_cutoff=logit_cutoff
        )

        elapsed = time.perf_counter() - st_t
        logger.info(f"Search completed. Elapsed: {elapsed:.1f} s")

        # Re-fit RT calibrator with full search results and save predicted RTs for re-scoring
        rt_calibrator = self._perform_rt_calibration(run, before_full_search=False)
        logger.info("RT calibration fitted")

        if self.state == SearchState.FIRST_SEARCH:
            result_manager.write_df(
                df=run.meta_df.select(pl.exclude("peak_start", "peak_stop")),
                key="meta_df",
            )
            result_manager.write_attr("lc_peak_width", self.lc_peak_width)
            result_manager.write_attr(
                "gradient_length_in_seconds", run.gradient_length_in_seconds
            )
            result_manager.write_attr("xic_peak_interval", self.lc_peak_width / 8.0)

        return result_manager
