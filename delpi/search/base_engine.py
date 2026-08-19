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

from pymsio import MassSpecData, ReaderFactory
from delpi.lcms.dda_run import DDARun
from delpi.lcms.dia_run import DIARun
from delpi.model.classifier import DelPiModel
from delpi.search.config import SearchConfig
from delpi.search.result_manager import ResultManager
from delpi.database.peptide_database import PeptideDatabase
from delpi.search.tda.tda_processor import TDAProcessor
from delpi.model.pmsm_scale import PeptideMultiSpectraMatchScaler
from delpi.model.rt_calibrator import RetentionTimeCalibrator
from delpi.search.tl.data_prep import TransferLearningDataPreparator
from delpi.search.search_state import SearchState
from delpi.search.progress.tracker import ProgressTracker
from delpi.search.progress.tqdm_tracker import TqdmProgressTracker
from delpi.search.progress.callback_tracker import CallbackProgressTracker
from delpi.utils.fdr import calculate_q_value
from delpi.utils.log_config import configure_logging
from delpi.constants import DEFAULT_Q_VALUE_CUTOFF
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
        model = DelPiModel.load(MODEL_DIR / f"delpi.{acq_method.lower()}.pth").to(
            self.device
        )
        model.transform = PeptideMultiSpectraMatchScaler()
        return model

    @staticmethod
    def _limit_numba_threads():
        """Limit numba thread count in the current (background) thread.

        Called as ``thread_initializer`` for :class:`Prefetcher` so that
        background ``find_peak_groups`` work uses only half of the available
        cores, leaving the rest free for GPU-feeding data pipelines.
        """
        import numba

        n = max(1, numba.config.NUMBA_NUM_THREADS // 2)
        numba.set_num_threads(n)

    def get_save_quant(self) -> bool:
        return (
            self.state >= SearchState.SECOND_SEARCH
            or not self.search_config.enable_transfer_learning
        )

    def get_logit_cutoff(self, base_cutoff: float, loose_delta: float = -3.0) -> float:
        return (
            base_cutoff + loose_delta
            if self.state >= SearchState.SECOND_SEARCH
            else base_cutoff
        )

    def next_state(self):
        self.state = SearchState(self.state + 1)

    def get_db_dir(self):
        if not self.search_config.enable_transfer_learning:
            return self.search_config.db_dir
        return (
            self.search_config.db_dir
            if self.state < SearchState.SECOND_SEARCH
            else self.search_config.refined_db_dir
        )

    def get_results_group_key(self):
        if not self.search_config.enable_transfer_learning:
            return "first_results"
        return (
            "first_results"
            if self.state < SearchState.SECOND_SEARCH
            else "second_results"
        )

    @abstractmethod
    def perform_quick_search(self, run: Union[DIARun, DDARun]) -> pl.DataFrame:
        pass

    @abstractmethod
    def perform_search(
        self, lcms_data: MassSpecData, progress: ProgressTracker = None
    ) -> ResultManager:
        """
        Perform the core search logic for a single LC-MS/MS run.

        This method contains the acquisition-specific search implementation
        and must be implemented by subclasses (DDA/DIA engines).

        Args:
            lcms_data: LC-MS/MS data
            progress: Progress tracker for reporting status

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

    def process_single_run(
        self,
        raw_path: Path,
        progress_queue=None,
    ) -> None:
        """Run the full search pipeline for a single raw file.

        This method is designed to be called in a child process spawned by
        :meth:`SearchManager.execute_batch`.

        Parameters
        ----------
        raw_path : Path
            Path to the raw LC-MS/MS file.
        progress_queue : multiprocessing.Queue, optional
            When provided, a :class:`CallbackProgressTracker` is created
            that puts :class:`ProgressSnapshot` objects onto the queue so
            the parent process can forward them.  A ``None`` sentinel is
            put on the queue when this method finishes (success or error).
            When *not* provided, a :class:`TqdmProgressTracker` is used
            for CLI output.
        """

        use_tqdm = progress_queue is None
        configure_logging(
            logfile_path=self.search_config.log_file_path,
            level=logging.INFO,
            use_tqdm=use_tqdm,
        )

        if progress_queue is not None:
            progress = CallbackProgressTracker(
                total=100,
                description="Overall",
                callback=lambda snap: progress_queue.put(snap),
            )
        else:
            progress = TqdmProgressTracker(total=100, description="Overall")

        try:
            logger.info(f"Loading LC-MS data: {raw_path}")
            # Load raw data (5% of overall)
            reader = ReaderFactory.get_reader(raw_path)
            num_scans = reader.num_spectra
            if num_scans is not None:
                load_progress = progress.create_child(
                    "Loading spectra", total=num_scans, portion=5
                )
                lcms_data = reader.load(progress=load_progress)
            else:
                load_progress = progress.create_child(
                    "Loading spectra", total=1, portion=5
                )
                lcms_data = reader.load()
            load_progress.complete()
            max_rt_time = lcms_data.meta_df.item(-1, "time_in_seconds")
            # lcms_data.meta_df.write_parquet(
            #     self.search_config.output_dir / f"{lcms_data.run_name}.meta_df.parquet"
            # )
            logger.info(
                f"Loaded {lcms_data.meta_df.shape[0]} spectra. Max scan time: {max_rt_time/60:.1f} min"
            )

            # Perform search (80% of overall)
            logger.info(f"Start {self.get_acquisition_method()} search")
            search_progress = progress.create_child("Search", total=100, portion=80)
            result_manager = self.perform_search(lcms_data, progress=search_progress)
            search_progress.complete()

            self.next_state()

            progress.complete()

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise RuntimeError(f"Search failed for {raw_path}") from e
        finally:
            progress.close()
            if progress_queue is not None:
                progress_queue.put(None)  # sentinel — tells parent we're done

    def prepare_tl_data_for_run(
        self,
        raw_path: Path,
        target_pmsm_df: pl.DataFrame,
        tl_ms2_h5_path: Path,
        progress_queue=None,
    ) -> None:
        """Extract MS2/RT transfer-learning training data for a single run
        and append it to the shared ``transfer_learning_ms2.h5`` file.

        This is run *after* the first-pass global (cross-run) target-decoy
        analysis, using the already dual-FDR-filtered and top-k-selected
        target PmSMs for this run (see
        :func:`delpi.search.tl.second_pass.select_tl_training_pmsms`).

        Designed to be called in a child process spawned by
        :meth:`SearchManager.prepare_transfer_learning_training_data`, for
        the same GPU-memory isolation and raw-data reload reasons as
        :meth:`process_single_run`.
        """
        use_tqdm = progress_queue is None
        configure_logging(
            logfile_path=self.search_config.log_file_path,
            level=logging.INFO,
            use_tqdm=use_tqdm,
        )

        try:
            if target_pmsm_df.shape[0] == 0:
                logger.info(f"No TL training PmSMs selected for {raw_path}; skipping")
                return

            preparator = TransferLearningDataPreparator(
                tolerance_in_ppm=self.search_config["ms2_mass_tol_in_ppm"],
                apply_phospho=self.search_config.is_phospho_search,
            )
            preparator.extract_and_save(target_pmsm_df, raw_path, tl_ms2_h5_path)
            logger.info(f"Transfer learning data prepared for {raw_path}")
        except Exception as e:
            logger.error(f"TL data prep failed: {str(e)}")
            raise RuntimeError(f"TL data prep failed for {raw_path}") from e
        finally:
            if progress_queue is not None:
                progress_queue.put(None)  # sentinel — tells parent we're done

    def perform_tda(self, result_manager: ResultManager) -> pl.DataFrame:
        """Run-specific TDA using TensorDataset for fast training/inference."""
        logger.info("Target-decoy analysis started")

        q_value_cutoff = self.search_config.config.get(
            "q_value_cutoff", DEFAULT_Q_VALUE_CUTOFF
        )
        use_protein_picker = self.search_config.config.get("use_protein_picker", True)
        grouping_type = self.search_config.config.get(
            "grouping_type", "parsimonious_grouping"
        )
        group_key = self.get_results_group_key()
        search_batch_size = self.search_config.config.get("batch_size", 512)
        processor = TDAProcessor(
            db_dir=self.get_db_dir(),
            output_dir=self.search_config.output_dir,
            device=self.device,
            q_value_cutoff=q_value_cutoff,
            use_protein_picker=use_protein_picker,
            grouping_type=grouping_type,
            batch_size=search_batch_size * 4,
        )
        pmsm_df = processor.run_single(result_manager, group_key)

        counts = ResultManager.compute_id_statistics(pmsm_df, q_value_cutoff)
        logger.info(
            "FDR estimated: "
            f"#Precursors: {counts['precursors']}, "
            f"#Peptides: {counts['peptides']}, "
            f"#Protein Groups: {counts['protein_groups']} "
            f"at {q_value_cutoff:.2f} FDR"
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
                results_dict = result_manager.read_dict(
                    group_key,
                    data_keys=[
                        "precursor_index",
                        "observed_rt",
                        "predicted_rt",
                        "score",
                        "precursor_q_value",
                    ],
                )
                pmsm_df = (
                    pl.DataFrame(results_dict)
                    .filter(pl.col("precursor_q_value").is_not_null())
                    .group_by("precursor_index")
                    .agg(pl.all().sort_by("score").last())
                )

                ## update precursor_index for refined DB
                precursor_df = (
                    pl.scan_parquet(db_dir / "precursor_df.parquet")
                    .select(pl.col("g_precursor_index", "precursor_index"))
                    .collect()
                )

                pmsm_df = pmsm_df.rename({"precursor_index": "g_precursor_index"}).join(
                    precursor_df.select(pl.col("g_precursor_index", "precursor_index")),
                    on="g_precursor_index",
                    how="left",
                )

                pmsm_df = PeptideDatabase.join(
                    db_dir,
                    pmsm_df,
                    precursor_columns=[],
                    modification_columns=["ref_rt"],
                    peptide_columns=["is_decoy"],
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
            min_rt_tolerance=0.1,
            max_rt_tolerance=0.15,
        )

        if after_full_search:
            # save RT predictions for the full search results
            rt_df = rt_calibrator.predict(pmsm_df["ref_rt"])
            result_manager.write_dict(
                group_key, {"predicted_rt": rt_df["predicted_rt"].to_numpy()}
            )

        return rt_calibrator
