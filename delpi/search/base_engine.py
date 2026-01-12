"""
Base Search Engine for DelPi

This module provides the base search engine class with common functionality
for both DDA and DIA search pipelines.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import torch
import polars as pl
import numpy as np

from delpi.lcms.dda_run import DDARun
from delpi.lcms.dia_run import DIARun
from delpi.model.classifier import DelPiModel
from delpi.search.config import SearchConfig
from delpi.lcms.reader.base import MassSpecData
from delpi.lcms.reader.factory import ReaderFactory
from delpi.search.result_aggregator import ResultsAggregator
from delpi.search.result_manager import ResultManager
from delpi.database.peptide_database import PeptideDatabase
from delpi.search.tda.dataset import DatasetSplitter, PMSMDataset
from delpi.search.tda.fdr_analyzer import FDRAnalyzer
from delpi.search.tda.trainer import TargetDecoyTrainer
from delpi.model.pmsm_scale import PeptideMultiSpectraMatchScaler
from delpi.model.rt_calibrator import RetentionTimeCalibrator
from delpi.search.tl.data_prep import (
    TransferLearningDataPreparator,
    TransferLearningConfig,
)
from delpi.search.search_state import SearchState
from delpi.utils.fdr import calculate_q_value
from delpi.utils.log_config import configure_logging
from delpi import MODEL_DIR


logger = logging.getLogger(__name__)


class BaseSearchEngine(ABC):
    """
    Abstract base search engine class containing common functionality for DDA and DIA searches.

    This class provides the foundation for peptide search engines with:
    - Configuration validation and device management
    - Target-decoy analysis pipeline
    - Process-based execution for stability
    - Result management and storage
    """

    def __init__(
        self,
        search_config: SearchConfig,
        device: torch.device,
        state: SearchState = SearchState.INIT,
    ):
        """
        Initialize the base search engine.

        Args:
            search_config: Configuration object containing search parameters
        """
        self.search_config = search_config
        self.device = device
        self.state = state

    def _load_model(self) -> DelPiModel:
        acq_method = self.get_acquisition_method()
        model = torch.load(
            MODEL_DIR / f"delpi.{acq_method.lower()}.pth",
            weights_only=False,
            map_location=self.device,
        )
        assert isinstance(model, DelPiModel)
        model.transform = PeptideMultiSpectraMatchScaler()
        return model.eval()

    def next_state(self):
        self.state = SearchState(self.state + 1)

    def get_db_dir(self):
        return (
            self.search_config.db_dir
            if self.state < SearchState.SECOND_SEARCH
            else self.search_config.refined_db_dir
        )

    def get_results_group_key(self):
        return (
            "first_results"
            if self.state < SearchState.SECOND_SEARCH
            else "second_results"
        )

    @abstractmethod
    def perform_quick_search(self, run: Union[DIARun, DDARun]) -> pl.DataFrame:
        pass

    @abstractmethod
    def perform_search(self, lcms_data: MassSpecData) -> ResultManager:
        """
        Perform the core search logic for a single LC-MS/MS run.

        This method contains the acquisition-specific search implementation
        and must be implemented by subclasses (DDA/DIA engines).

        Args:
            search_config: Search configuration
            lcms_data: LC-MS/MS data

        Returns:
            ResultManager with search results
        """
        pass

    @abstractmethod
    def get_acquisition_method(self) -> str:
        """
        Get the acquisition method (DDA or DIA).

        Returns:
            Acquisition method string
        """
        pass

    def empty_device_cache(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def process_single_run(self, raw_path: Path) -> None:

        configure_logging(
            logfile_path=self.search_config.log_file_path, level=logging.INFO
        )

        try:
            logger.info(f"Loading LC-MS data: {raw_path}")
            # Load raw data
            # reader = ReaderFactory.get_reader(
            #     raw_path, bmsio_server_addr=self.search_config.bmsio_server_addr
            # )
            reader = ReaderFactory.get_reader(
                raw_path, acquisition_method=self.get_acquisition_method()
            )
            lcms_data = reader.load()
            max_rt_time = lcms_data.meta_df.item(-1, "time_in_seconds")
            logger.info(
                f"Loaded {lcms_data.meta_df.shape[0]} spectra. Max scan time: {max_rt_time/60:.1f} min"
            )

            # Perform search (implemented by subclasses)
            logger.info(f"Start {self.get_acquisition_method()} search")
            result_manager = self.perform_search(lcms_data)

            self.next_state()
            if self.state < SearchState.SECOND_SEARCH:
                pmsm_df = self.perform_tda(result_manager)

                group_key = self.get_results_group_key()
                result_manager.write_df(pmsm_df, key=f"{group_key}/pmsm_df")
                self.next_state()
                tl_dataset = self.prepare_transfer_learning_data(
                    lcms_data,
                    pmsm_df.filter(pl.col("is_decoy") == False),
                    mass_tolerance_in_ppm=self.search_config["ms2_mass_tol_in_ppm"],
                    is_phospho_search=self.search_config.is_phospho_search,
                )
                result_manager.write_tl_data(tl_dataset)
                logger.info("Transfer learning data prepared")

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise RuntimeError(f"Search failed for {raw_path}") from e

    def perform_tda(self, result_manager: ResultManager) -> pl.DataFrame:

        logger.info("Target-decoy analysis started")
        is_dia = self.get_acquisition_method().lower() == "dia"
        pmsm_sel_col = "apex_median_intensity" if is_dia else "score"

        # Extract parameters from config
        q_value_cutoff = self.search_config.config["q_value_cutoff"]

        # Load search results
        group_key = self.get_results_group_key()
        result_aggregator = ResultsAggregator(
            db_dir=self.get_db_dir()
        ).add_result_manager(0, result_manager)

        pmsm_df, data_dict = result_aggregator.get_search_results(
            group_key, load_features=True, estimate_apex_intensity=is_dia
        )

        num_decoys = pmsm_df["is_decoy"].sum()
        num_targets = len(pmsm_df) - num_decoys
        logger.info(
            f"Training a classifier with {num_targets:,} positive and {num_decoys:,} negative PMSMs"
        )

        # Train classifier and rescore PMSMs
        splitter = DatasetSplitter()
        train_df, test_df = splitter.split_by_peptide(pmsm_df)
        train_dataset = PMSMDataset.create_tensor_dataset(train_df, data_dict)
        test_dataset = PMSMDataset.create_tensor_dataset(test_df, data_dict)
        trainer = TargetDecoyTrainer()
        test_score_arr = trainer.train(
            result_manager.run_name,
            train_dataset,
            test_dataset,
            output_dir=self.search_config.output_dir,
            device=self.device,
        )

        ###############################################################################
        # select only one PMSM per cluster (sharing the same peaks)
        # then select only one PMSM per precursor
        # the order of these two steps does matter
        # because low-scoring PMSMs of precursor may not share peaks with other PMSMs
        ################################################################################
        pmsm_df = (
            test_df.with_columns(score=test_score_arr)
            .group_by(["cluster"])
            .agg(pl.all().sort_by("score").last())
        )
        logger.info(
            f"Selected {pmsm_df.shape[0]} non-redundant PMSMs (one per cluster) from {test_df.shape[0]} PMSMs"
        )

        ## Now select only one PMSM per precursor
        ## Use apex_median_intensity instead of score to avoid PMSMs in XIC tails

        pmsm_df = pmsm_df.group_by(["precursor_index"]).agg(
            pl.all().sort_by(pmsm_sel_col).last()
        )

        logger.debug(
            f"Selected {pmsm_df.shape[0]} best-scoring PMSMs (one per Precursor)"
        )

        fdr_analyzer = FDRAnalyzer(
            q_value_cutoff=q_value_cutoff, db_dir=self.search_config.db_dir
        )
        pmsm_df = fdr_analyzer.perform_run_specific_analysis(pmsm_df)
        counts = ResultManager.compute_id_statistics(pmsm_df, q_value_cutoff)
        logger.info(
            "FDR estimated: "
            f"#Precursors: {counts['precursors']}, "
            f"#Peptides: {counts['peptides']}, "
            f"#Protein Groups: {counts['protein_groups']} at {q_value_cutoff:.2f} FDR"
        )

        return pmsm_df.filter(pl.col("precursor_q_value") <= q_value_cutoff)

    def _perform_rt_calibration(
        self, run: Union[DIARun, DDARun], before_full_search: bool
    ):
        after_full_search = not before_full_search
        result_manager = self.search_config.get_result_manager(run.name)
        meta_df = run.meta_df

        if isinstance(run, DIARun):
            if before_full_search:
                fig_path = self.search_config.output_dir / f"{run.name}_rt_calib_1.jpg"
            else:
                fig_path = self.search_config.output_dir / f"{run.name}_rt_calib_2.jpg"
        else:
            fig_path = self.search_config.output_dir / f"{run.name}_rt_calib.jpg"

        ## Before second full search, use first search results with original DB
        if before_full_search:
            group_key = "first_results"
            #### refined DB contains ref_rt predicted by fine-tuned RT model
            # db_dir = self.search_config.db_dir
            db_dir = self.get_db_dir()
        else:
            group_key = self.get_results_group_key()
            db_dir = self.get_db_dir()

        if self.state == SearchState.FIRST_SEARCH and before_full_search:
            q_value_cutoff = 0.05
            pmsm_df = self.perform_quick_search(run)
            target_df = pmsm_df.filter(pl.col("is_decoy") == False)
        else:
            q_value_cutoff = 0.01
            if after_full_search:
                results_dict = result_manager.read_dict(
                    group_key, data_keys=["precursor_index", "observed_rt", "logit"]
                )
                pmsm_df = pl.DataFrame(results_dict)
                pmsm_df = PeptideDatabase.join(
                    db_dir,
                    pmsm_df,
                    precursor_columns=[],
                    modification_columns=["ref_rt"],
                    peptide_columns=["is_decoy"],
                )
                pmsm_df = calculate_q_value(
                    pmsm_df, score_column="logit", out_column="precursor_q_value"
                )
            else:  # SECND_SEARCH and before_full_search
                pmsm_df = result_manager.read_df(f"{group_key}/pmsm_df")

                ## update precursor_index for refined DB
                precursor_df = (
                    pl.scan_parquet(db_dir / "precursor_df.parquet")
                    .select(pl.col("g_precursor_index", "precursor_index"))
                    .collect()
                )
                req_cols = pl.col(
                    "precursor_index",
                    "observed_rt",
                    "predicted_rt",
                    "is_decoy",
                    "precursor_q_value",
                )

                pmsm_df = (
                    pmsm_df.select(req_cols)
                    .rename({"precursor_index": "g_precursor_index"})
                    .join(
                        precursor_df.select(
                            pl.col("g_precursor_index", "precursor_index")
                        ),
                        on="g_precursor_index",
                        how="left",
                    )
                )

                pmsm_df = PeptideDatabase.join(
                    db_dir,
                    pmsm_df,
                    precursor_columns=[],
                    modification_columns=["ref_rt"],
                    peptide_columns=[],
                )

            target_df = pmsm_df.filter(
                (pl.col("precursor_q_value") <= q_value_cutoff)
                & (pl.col("is_decoy") == False)
            )

        rt_calibrator = RetentionTimeCalibrator.train(
            min_rt_in_seconds=meta_df.item(0, "time_in_seconds"),
            max_rt_in_seconds=meta_df.item(-1, "time_in_seconds"),
            ref_rt=target_df["ref_rt"].to_numpy(),
            obs_rt=target_df["observed_rt"].to_numpy(),
            degree=5 if self.state < SearchState.SECOND_SEARCH else 2,
            # figure_path=fig_path,
        )

        if after_full_search:
            # save RT predictions for the full search results
            rt_df = rt_calibrator.predict(pmsm_df["ref_rt"])
            result_manager.write_dict(
                group_key, {"predicted_rt": rt_df["predicted_rt"].to_numpy()}
            )

        return rt_calibrator

    def prepare_transfer_learning_data(
        self,
        lcms_data: MassSpecData,
        pmsm_df: pl.DataFrame,
        is_phospho_search: bool = False,
        mass_tolerance_in_ppm: float = 10,
    ):
        config = TransferLearningConfig(
            tolerance_in_ppm=mass_tolerance_in_ppm, apply_phospho=is_phospho_search
        )
        preparator = TransferLearningDataPreparator(config)
        collected_data = preparator.prepare_training_data(pmsm_df, lcms_data)

        return collected_data
