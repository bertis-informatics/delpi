"""
Search Coordinator for DelPi

This module provides the main search workflow coordination and factory for creating
acquisition-specific search engines.
"""

import logging
import time
import yaml
import queue
from pathlib import Path

import polars as pl
import numpy as np
import torch
from lightning.pytorch.accelerators import AcceleratorRegistry
from tabulate import tabulate

from pymsio import MassSpecFileReader
from delpi.search.config import SearchConfig
from delpi.search.base_engine import BaseSearchEngine
from delpi.search.dda.engine import DDASearchEngine
from delpi.search.dia.engine import DIASearchEngine
from delpi.search.result_aggregator import ResultsAggregator
from delpi.search.dia.lfq import LabelFreeQuantifier
from delpi.search.result_manager import ResultManager
from delpi.search.tl.rt_trainer import TransferLearningTrainerForRT
from delpi.search.tl.trainer import TransferLearningTrainer
from delpi.search.tl.spec_lib_generator import RefinedSpectralLibGenerator
from delpi.database.peptide_database import PeptideDatabase
from delpi.search.tda.tda_processor import TDAProcessor
from delpi.search.tda.fdr_analyzer import FDRAnalyzer
from delpi.search.search_state import SearchState
from delpi.search.progress import CallbackProgressTracker
from delpi.search.tl.second_pass import (
    select_tl_training_pmsms,
    select_second_pass_targets,
    select_paired_decoys,
)
from delpi.search.pmsm_assignment import assign_pmsms_across_runs
from delpi.search.dia.max_lfq import maxlfq
from delpi.utils.mp import get_multiprocessing_context
from delpi.database.utils import get_modified_sequence
from delpi.constants import (
    DEFAULT_Q_VALUE_CUTOFF,
    DEFAULT_REPORT_FORMAT,
    DEFAULT_TL_TOP_K,
)

SUPPORTED_DEVICES = ["cuda"]

logger = logging.getLogger(__name__)


class SearchManager:
    """
    Main search workflow manager for DelPi peptide identification.

    This class serves as the primary interface for peptide search workflows:
    - Factory pattern for creating acquisition-specific engines (DDA/DIA)
    - Coordinates multi-run search execution with process isolation
    - Manages cross-run quantification and result aggregation
    - Generates comprehensive search reports and statistics

    The manager automatically creates the appropriate search engine based on the
    'acquisition_method' configuration parameter.
    """

    def __init__(
        self,
        search_config: SearchConfig,
        specified_device: str = "auto",
        progress: CallbackProgressTracker = None,
    ):
        """
        Initialize the search coordinator.

        Args:
            search_config: Configuration object containing search parameters
            specified_device: Device specification (e.g. 'cuda:0', 'auto')
            progress: Optional CallbackProgressTracker.  When provided,
                a ``multiprocessing.Queue`` bridge is used to forward
                :class:`ProgressSnapshot` objects from the child process
                to this tracker's callback.  ``process_single_run`` still
                runs in a separate process for GPU-memory isolation.
                When ``None`` (default), each child process shows a tqdm
                progress bar.
        """
        self.search_config: SearchConfig = search_config
        self._validated_device: torch.device = None
        self._progress: CallbackProgressTracker = progress
        self.state: SearchState = SearchState.INIT
        self.check_device(specified_device)

    @property
    def output_dir(self) -> Path:
        """Output directory path."""
        return self.search_config.output_dir

    @property
    def log_file_path(self) -> Path:
        """Log file path."""
        return self.search_config.log_file_path

    @property
    def input_files(self) -> list:
        """Input raw files list."""
        return self.search_config.input_files

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

    @property
    def device(self) -> torch.device:
        """Validated device for computation."""
        return self._validated_device

    def get_engine(self) -> BaseSearchEngine:
        """
        Create and return a search engine instance.

        Returns:
            BaseSearchEngine subclass instance
        """
        # Determine acquisition method from config
        acq_method = self.search_config.config.get("acquisition_method", "DIA")

        if acq_method.upper() == "DDA":
            return DDASearchEngine(self.search_config, self.device, self.state)
        elif acq_method.upper() == "DIA":
            return DIASearchEngine(self.search_config, self.device, self.state)
        else:
            raise ValueError(
                f"Unsupported acquisition method: {acq_method}. "
                "Supported methods: 'DDA', 'DIA'"
            )

    def prepare_database(self) -> None:
        """Prepare the peptide database if it doesn't exist."""
        self.state = SearchState.DB_PREP
        if not self.search_config.check_database_exists():
            if self.search_config.config.get("fasta_file") is None:
                raise ValueError("FASTA file is not specified in configuration")

            from delpi.search.database import build_database_in_subprocess

            build_database_in_subprocess(self.search_config, self.device)
        else:
            logger.info(f"Use existing peptide database: {self.search_config.db_dir}")

    def check_device(self, specified_device: str = "auto") -> None:
        """
        Check and validate device configuration.

        This method:
        1. Checks if device is specified in search_config
        2. Validates the specified device is available on the current machine
        3. Falls back to automatic device detection if not specified
        4. Logs the final device configuration

        Raises:
            RuntimeError: If the specified device is not available
        """

        if specified_device is not None and specified_device.lower() != "auto":
            try:
                device = torch.device(specified_device.lower())
            except RuntimeError as e:
                raise RuntimeError(
                    f"Invalid device specification '{specified_device}': {e}"
                )

            if device.type not in SUPPORTED_DEVICES:
                raise RuntimeError(
                    f"Unsupported device type: {device.type}. Supported devices: {SUPPORTED_DEVICES}"
                )
            accelerator = AcceleratorRegistry.get(device.type)
            if not accelerator.is_available():
                raise RuntimeError(
                    f"'{device.type}' was specified but is not available on this machine"
                )
            else:
                logger.info(
                    f"Specified device '{specified_device}' is available and will be used"
                )
        else:
            for device_type in SUPPORTED_DEVICES:
                accelerator = AcceleratorRegistry.get(device_type)
                if accelerator.is_available():
                    device = torch.device(device_type)
                    break
            logger.info(f"Auto-detected device: {device}")

        # Log device information
        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(device)
            device_memory = torch.cuda.get_device_properties(device).total_memory / (
                1024**3
            )
            logger.info(f"Using CUDA: {device_name} with {device_memory:.1f} GB memory")
        elif device.type == "mps":
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            logger.info("Using CPU")

        # Store the validated device for use by engines
        self._validated_device = device

        # Resolve batch_size if set to 'auto'
        self._resolve_batch_size()

    def _resolve_batch_size(self) -> None:
        """Resolve ``batch_size`` in search config from 'auto' or an explicit value.

        Rule of thumb: 1024 for 24 GB GPU, scaling linearly and rounding
        down to the nearest power of 2.  Clamped to [256, 2048].
        """
        raw = self.search_config.config.get("batch_size", "auto")
        if isinstance(raw, int) or (isinstance(raw, str) and raw.isdigit()):
            self.search_config.config["batch_size"] = int(raw)
            return

        # auto – determine from GPU memory
        device = self._validated_device
        if device is not None and device.type == "cuda":
            mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        else:
            mem_gb = 12  # conservative fallback

        raw_bs = mem_gb / 24 * 1024
        # round to nearest power of 2
        log2 = (
            raw_bs.bit_length() - 1
            if isinstance(raw_bs, int)
            else int(raw_bs).bit_length() - 1
        )
        lower = 1 << log2
        upper = 1 << (log2 + 1)
        bs = lower if (raw_bs - lower) < (upper - raw_bs) else upper
        bs = max(256, min(bs, 2048))
        self.search_config.config["batch_size"] = bs
        logger.info(f"Auto-resolved batch size: {bs}")

    def execute_batch(self) -> None:
        """Execute workflow for all input files using separate processes.

        Each run is **always** executed in a child process for GPU-memory
        isolation.  When a :class:`CallbackProgressTracker` was supplied at
        construction time, a ``multiprocessing.Queue`` bridges progress
        snapshots from the child back to the parent so that the user's
        callback fires in-process.
        """

        input_files = self.input_files

        if self.state < SearchState.FIRST_SEARCH:
            self.state = SearchState.FIRST_SEARCH
        else:
            self.state = SearchState.SECOND_SEARCH
            logger.info(f"Second search after transfer learning")

        logger.info(f"Total runs to process: {len(input_files)}")

        # Create engine instance for process execution
        engine = self.get_engine()
        mp_ctx = get_multiprocessing_context()

        for run_idx, raw_path in enumerate(input_files):
            run_name = MassSpecFileReader.extract_run_name(raw_path)
            st_t = time.perf_counter()
            logger.info(f"[{run_idx+1}/{len(input_files)}] Processing run: {run_name}")

            # Set up queue bridge when an external tracker is provided
            progress_queue = mp_ctx.Queue() if self._progress is not None else None

            p = mp_ctx.Process(
                target=engine.process_single_run,
                args=(raw_path,),
                kwargs={"progress_queue": progress_queue} if progress_queue else {},
            )
            p.start()
            logger.debug(f"Start a child process (PID: {p.pid})")

            if progress_queue is not None:
                # Drain snapshots from the child and forward to the user's tracker
                self._drain_progress_queue(progress_queue, p)

            p.join()

            if p.exitcode != 0:
                raise RuntimeError(f"Searching failed with exit code {p.exitcode}")
            logger.debug(f"Terminate child process (PID: {p.pid})")

            elapsed = time.perf_counter() - st_t
            logger.info(
                f"[{run_idx+1}/{len(input_files)}] Completed processing. Elapsed: {elapsed:.1f} s"
            )

    def _drain_progress_queue(self, progress_queue, process) -> None:
        """Read :class:`ProgressSnapshot` objects from *progress_queue* until
        the child sends a ``None`` sentinel or the child process exits."""
        while True:
            try:
                snapshot = progress_queue.get(timeout=1.0)
            except queue.Empty:
                if not process.is_alive():
                    break
                continue
            if snapshot is None:
                break
            self._progress.forward_snapshot(snapshot)

    def perform_transfer_learning(self, first_pmsm_df: pl.DataFrame) -> None:
        """Fine-tune the RT/MS2 predictors and build the refined (second-pass)
        target-decoy library from the first-pass global scoring results.

        Parameters
        ----------
        first_pmsm_df:
            Output of ``perform_global_tda(SearchState.FIRST_TDA, ...)`` — the
            first-pass PmSM DataFrame with run-specific and global q-values
            (and, since it is the first pass, protein grouping).
        """

        self.state = SearchState.TL_TRAINING
        output_dir = self.search_config.output_dir
        search_config = self.search_config
        top_k = search_config.config.get("tl_top_k", DEFAULT_TL_TOP_K)
        # The cutoff used to select PmSMs for predictor fine-tuning is fixed
        # (not user-configurable) so that fine-tuning quality is decoupled
        # from whatever (possibly looser) q_value_cutoff the user chose for
        # reporting.
        tl_q_value_cutoff = 0.01

        logger.info("Transfer learning started")

        logger.info("Selecting target PmSMs for predictor fine-tuning")
        tl_training_df = select_tl_training_pmsms(
            first_pmsm_df, q_value_cutoff=tl_q_value_cutoff, top_k=top_k
        )
        self.prepare_transfer_learning_training_data(tl_training_df)

        result_aggregator = ResultsAggregator(
            db_dir=self.get_db_dir(), search_config=search_config
        )
        tl_ms2_h5_path = search_config.tl_ms2_h5_path
        logger.info("Training RT predictor")
        rt_trainer = TransferLearningTrainerForRT()
        rt_predictor = rt_trainer.train(
            output_dir=output_dir,
            result_aggregator=result_aggregator,
            device=self.device,
            tl_ms2_h5_path=tl_ms2_h5_path,
        )
        del rt_trainer

        logger.info("Training MS2 spectrum predictor")
        trainer = TransferLearningTrainer()
        ms2_predictor = trainer.train(
            output_dir=output_dir,
            result_aggregator=result_aggregator,
            device=self.device,
            tl_ms2_h5_path=tl_ms2_h5_path,
        )
        del trainer

        return rt_predictor, ms2_predictor

    def prepare_transfer_learning_training_data(
        self, tl_training_df: pl.DataFrame
    ) -> None:
        """Reload each run's raw file and extract MS2/RT training data for the
        dual-FDR-filtered, top-k-selected target PmSMs (per run) chosen by
        :func:`~delpi.search.tl.second_pass.select_tl_training_pmsms`."""

        logger.info("Preparing transfer learning training data per run")
        engine = self.get_engine()
        mp_ctx = get_multiprocessing_context()
        tl_ms2_h5_path = self.search_config.tl_ms2_h5_path

        for run_idx, raw_path in enumerate(self.input_files):
            run_name = MassSpecFileReader.extract_run_name(raw_path)
            run_target_df = tl_training_df.filter(pl.col("run_index") == run_idx)
            if run_target_df.shape[0] == 0:
                logger.info(f"No TL training PmSMs selected for run {run_name}")
                continue

            progress_queue = mp_ctx.Queue() if self._progress is not None else None
            p = mp_ctx.Process(
                target=engine.prepare_tl_data_for_run,
                args=(raw_path, run_target_df, tl_ms2_h5_path),
                kwargs={"progress_queue": progress_queue} if progress_queue else {},
            )
            p.start()

            if progress_queue is not None:
                self._drain_progress_queue(progress_queue, p)

            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"TL data prep failed for run {run_name} with exit code {p.exitcode}"
                )

    def build_refined_library(
        self,
        first_pmsm_df: pl.DataFrame,
        rt_predictor,
        ms2_predictor,
    ) -> None:
        """Build the second-pass refined target-decoy library.

        Selects the unique target precursors confirmed by the first-pass
        global FDR cutoff, pairs each with exactly one decoy precursor, and
        generates the refined spectral library with the fine-tuned RT/MS2
        predictors. Also persists ``library_confidence.parquet`` so the
        second pass can reuse the first pass's protein grouping and identify
        precursor-run pairs by library + run-specific FDR (see
        :meth:`perform_global_tda` / :meth:`FDRAnalyzer.perform_global_analysis`).
        """

        self.state = SearchState.REFINED_DB_PREP

        search_config = self.search_config
        q_value_cutoff = search_config.config.get(
            "q_value_cutoff", DEFAULT_Q_VALUE_CUTOFF
        )

        logger.info("Selecting second-pass target precursors")
        target_df = select_second_pass_targets(
            first_pmsm_df, q_value_cutoff=q_value_cutoff
        )

        logger.info(f"Selecting paired decoys for {target_df.shape[0]:,} targets")
        target_df = select_paired_decoys(search_config.db_dir, target_df)
        target_df = target_df.filter(pl.col("decoy_precursor_index").is_not_null())

        combined_precursor_index_arr = np.unique(
            np.concatenate(
                [
                    target_df["precursor_index"].to_numpy(),
                    target_df["decoy_precursor_index"].to_numpy(),
                ]
            )
        ).astype(np.uint32)

        logger.info(
            f"Generating refined spectral library with fine-tuned models "
            f"({target_df.shape[0]:,} targets + paired decoys)"
        )
        spec_generator = RefinedSpectralLibGenerator(
            apply_phospho=self.search_config.is_phospho_search,
            min_precursor_charge=search_config["precursor"].get("min_charge", 2),
            max_precursor_charge=search_config["precursor"].get("max_charge", 4),
            min_precursor_mz=search_config["precursor"].get("min_mz", 300),
            max_precursor_mz=search_config["precursor"].get("max_mz", 1800),
            min_fragment_charge=search_config["fragment"].get("min_charge", 1),
            max_fragment_charge=search_config["fragment"].get("max_charge", 2),
            min_fragment_mz=search_config["fragment"].get("min_mz", 200),
            max_fragment_mz=search_config["fragment"].get("max_mz", 1800),
            ms2_predictor=ms2_predictor,
            rt_predictor=rt_predictor,
        )

        spec_generator.generate_spectral_lib(
            search_config.db_dir,
            combined_precursor_index_arr,
        )
        spec_generator.save(search_config.refined_db_dir)

        self.save_library_confidence(target_df, spec_generator)

    def save_library_confidence(
        self,
        target_df: pl.DataFrame,
        spec_generator: RefinedSpectralLibGenerator,
    ) -> None:
        """Persist first-pass identification confidence for each second-pass
        target, keyed by the refined library's local ``precursor_index`` so
        the second pass can join directly on ``precursor_index``."""

        library_confidence_df = (
            target_df.select(
                pl.col(
                    "precursor_index",
                    "fasta_id",
                    "protein_index",
                    "protein_group",
                    "master_protein",
                    "global_precursor_q_value",
                    "global_peptide_q_value",
                    "global_protein_group_q_value",
                )
            )
            .rename(
                {
                    "precursor_index": "g_precursor_index",
                    "global_precursor_q_value": "library_precursor_q_value",
                    "global_peptide_q_value": "library_peptide_q_value",
                    "global_protein_group_q_value": "library_protein_group_q_value",
                }
            )
            .join(
                spec_generator.precursor_df.select(
                    pl.col("precursor_index", "peptidoform_index", "g_precursor_index")
                ),
                on="g_precursor_index",
                how="inner",
            )
            .join(
                spec_generator.modification_df.select(
                    pl.col("peptidoform_index", "peptide_index")
                ),
                on="peptidoform_index",
                how="left",
            )
            .select(
                pl.col(
                    "peptide_index",
                    "peptidoform_index",
                    "precursor_index",
                    "fasta_id",
                    "protein_index",
                    "protein_group",
                    "master_protein",
                    "library_precursor_q_value",
                    "library_peptide_q_value",
                    "library_protein_group_q_value",
                )
            )
        )

        refined_db_dir = self.search_config.refined_db_dir
        refined_db_dir.mkdir(parents=True, exist_ok=True)
        library_confidence_df.write_parquet(
            refined_db_dir / "library_confidence.parquet"
        )

    def _apply_fdr(
        self,
        pmsm_df: pl.DataFrame,
        result_aggregator: ResultsAggregator,
        q_value_cutoff: float,
        use_protein_picker: bool,
        grouping_type: str,
        run_protein_grouping: bool,
        library_confidence_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Run FDR control -- a separate concern from scoring/assignment (TDAProcessor)."""
        fdr = FDRAnalyzer(
            q_value_cutoff=q_value_cutoff,
            db_dir=self.get_db_dir(),
            use_protein_picker=use_protein_picker,
            grouping_type=grouping_type,
        )
        pmsm_df = fdr.perform_global_analysis(
            pmsm_df,
            protein_inference=run_protein_grouping,
            library_confidence_df=library_confidence_df,
        )
        pmsm_df = fdr.batch_run_specific_analysis(pmsm_df)
        return pmsm_df.join(result_aggregator.get_run_df(), on="run_index", how="left")

    def perform_global_tda(
        self,
        state: SearchState,
        run_protein_grouping: bool = True,
    ) -> pl.DataFrame:
        """Run cross-run (global) target-decoy analysis.

        Parameters
        ----------
        state:
            The :class:`SearchState` to transition to before running (e.g.
            ``FIRST_TDA`` for the first pass, ``SECOND_TDA`` for the second
            pass). This determines which results group / database directory
            is used (see :meth:`get_results_group_key`, :meth:`get_db_dir`).
        run_protein_grouping:
            Whether to perform protein inference/grouping from scratch.
            Only used for the first pass; the second pass instead reuses
            the first pass's target protein grouping (and freshly groups
            decoys) via ``library_confidence.parquet`` — see
            :meth:`FDRAnalyzer.perform_global_analysis`.
        """
        self.state = state
        logger.info("Performing global target-decoy analysis")

        search_config = self.search_config
        group_key = self.get_results_group_key()
        q_value_cutoff = search_config.config.get(
            "q_value_cutoff", DEFAULT_Q_VALUE_CUTOFF
        )
        use_protein_picker = search_config.config.get("use_protein_picker", True)
        grouping_type = search_config.config.get(
            "grouping_type", "parsimonious_grouping"
        )

        result_aggregator = ResultsAggregator(
            db_dir=self.get_db_dir(), search_config=search_config
        )

        # Second pass: reuse the first pass's target protein grouping
        # (and freshly group decoys) instead of running protein inference
        # from scratch — see FDRAnalyzer.perform_global_analysis.
        library_confidence_df = None
        if state == SearchState.SECOND_TDA:
            library_confidence_df = pl.read_parquet(
                search_config.refined_db_dir / "library_confidence.parquet"
            )

        search_batch_size = search_config.config.get("batch_size", 512)
        processor = TDAProcessor(
            db_dir=self.get_db_dir(),
            output_dir=search_config.output_dir,
            device=self.device,
            q_value_cutoff=q_value_cutoff,
            use_protein_picker=use_protein_picker,
            grouping_type=grouping_type,
            batch_size=search_batch_size * 4,
            split_level="peptide",
        )

        # 1) score every PmSM
        scored_df = processor.run_global(
            result_aggregator,
            group_key,
            training_params={
                "num_warmup_steps": 5,
                "max_epochs": 50,
                "train_split": 0.8,
                "early_stopping_patience": 5,
            },
            pass_label="first" if state < SearchState.SECOND_SEARCH else "second",
        )

        # 2) assign one PmSM per run/precursor (score + median intensity + alignment-group DP)
        intensity_weight = self.search_config.config.get("intensity_weight", 4.0)
        pmsm_df = assign_pmsms_across_runs(scored_df, intensity_weight=intensity_weight)
        pmsm_df = PeptideDatabase.join_with_protein_annotations(
            result_aggregator.db_dir, pmsm_df
        )

        # 3) FDR control
        pmsm_df = self._apply_fdr(
            pmsm_df,
            result_aggregator,
            q_value_cutoff,
            use_protein_picker,
            grouping_type,
            run_protein_grouping,
            library_confidence_df,
        )

        if state == SearchState.FIRST_TDA:
            # Persist score/precursor_q_value back into each run's own
            # "first_results" HDF group so that the RT calibration bootstrap
            # ahead of the second full search (see
            # BaseSearchEngine._perform_rt_calibration) can reuse them
            # without a redundant run-specific TDA pass.
            processor.write_back_scores(pmsm_df, result_aggregator, group_key)

        # Second pass: report/aggregate using the first-pass-derived library
        # q-values instead of this pass's own (diagnostic-only) global
        # q-value, since only library-confirmed precursors are reportable.
        # (protein_group/master_protein and library_*_q_value columns are
        # already joined onto pmsm_df above, via FDRAnalyzer.perform_global_analysis.)
        self.log_id_statistics_table(
            pmsm_df,
            q_value_cutoff,
            use_library_q_value=library_confidence_df is not None,
        )
        return pmsm_df

    def perform_quantification(
        self,
        pmsm_df: pl.DataFrame,
        library_q_value_column: str = "global_precursor_q_value",
    ) -> None:

        logger.info("Performing cross-run quantification")
        self.state = SearchState.QUANTIFICATION
        q_value_cutoff = self.search_config.config.get(
            "q_value_cutoff", DEFAULT_Q_VALUE_CUTOFF
        )

        result_aggregator = ResultsAggregator(
            db_dir=self.get_db_dir(), search_config=self.search_config
        )

        # FDR filtering is this caller's responsibility, not LabelFreeQuantifier's
        # (it only quantifies whatever it's given): only PmSMs passing both the
        # run-specific and library-level cutoffs are handed to LFQ.
        target_pmsm_df = pmsm_df.filter(
            (pl.col("is_decoy") == False)
            & (pl.col(library_q_value_column) <= q_value_cutoff)
            & (pl.col("precursor_q_value") <= q_value_cutoff)
        )

        lfq = LabelFreeQuantifier(
            result_aggregator,
            group_key=self.get_results_group_key(),
            acq_method=self.search_config.config.get("acquisition_method", "DDA"),
        )

        # LabelFreeQuantifier returns a minimal DataFrame keyed by
        # (run_index, precursor_index) with just the new quant columns; only
        # rows passing the FDR filter above were quantified, so this must be
        # joined back onto the full pmsm_df (per-run assignment upstream
        # already guarantees at most one PmSM per (run_index, precursor_index),
        # so a plain left join is all that's needed -- no re-selection).
        quant_df = lfq.perform_quantification(target_pmsm_df)
        value_columns = [
            c for c in quant_df.columns if c not in ("run_index", "precursor_index")
        ]
        pmsm_df = pmsm_df.select(pl.exclude(value_columns)).join(
            quant_df, on=["run_index", "precursor_index"], how="left"
        )

        ## run MaxLFQ
        if self.search_config.config.get("acquisition_method", "DDA").upper() == "DIA":
            logger.info("Performing protein quantification with MaxLFQ ")
            protein_group_q_value_column = (
                "library_protein_group_q_value"
                if library_q_value_column == "library_precursor_q_value"
                else "global_protein_group_q_value"
            )
            df = (
                pmsm_df.filter(pl.col("is_decoy") == False)
                .filter(pl.col(protein_group_q_value_column) <= q_value_cutoff)
                .filter(
                    pl.col("ms2_quantity_normalized").is_not_null()
                    & (pl.col("ms2_quantity_normalized") > 0)
                )
            )
            # protein-group abundance is computed from run/RT-normalized
            # precursor quantities, not the raw ms2_quantity.
            pg_quant_df = maxlfq(
                df,
                min_peptides_per_protein=1,
                peptide_col="precursor_index",
                intensity_col="ms2_quantity_normalized",
            )
            pg_quant_df = pg_quant_df.join(
                result_aggregator.get_run_df(), on="run_index", how="left"
            )
        else:
            pg_quant_df = None

        return pmsm_df, pg_quant_df

    @staticmethod
    def _add_report_columns(pmsm_df: pl.DataFrame) -> pl.DataFrame:
        """Add ``modified_sequence`` and ``posterior_error`` report columns."""
        return pmsm_df.with_columns(
            pl.col("peptide_index", "peptidoform_index"),
            pl.when(pl.col("mod_ids").is_null())
            .then(pl.col("peptide"))
            .otherwise(
                pl.struct(["peptide", "mod_ids", "mod_sites"]).map_elements(
                    lambda x: get_modified_sequence(
                        x["peptide"],
                        x["mod_ids"],
                        x["mod_sites"],
                        use_unimod_id=True,
                    ),
                    return_dtype=pl.String,
                )
            )
            .alias("modified_sequence"),
        ).with_columns(posterior_error=1 - (1 / (1 + (-pl.col("score")).exp())))

    def save_pmsm_df(
        self,
        pmsm_df: pl.DataFrame,
        pg_quant_df: pl.DataFrame = None,
        library_q_value_column: str = "global_precursor_q_value",
        two_pass_mode: bool = False,
        filename_stem: str = "pmsm_results",
    ) -> None:
        """Filter, annotate and write PmSM identification results (and,
        when provided, MaxLFQ protein-group quantification) to disk.

        Also used for the first-pass-only identification snapshot (before
        the second pass / quantification), by passing
        ``filename_stem="pmsm_results.first"`` and leaving *pg_quant_df*
        unset — see :meth:`execute_workflow`.
        """

        format = self.search_config.config.get(
            "output_format", DEFAULT_REPORT_FORMAT
        ).lower()
        output_decoy = self.search_config.config.get("output_decoy", True)
        q_value_cutoff = self.search_config.config.get(
            "q_value_cutoff", DEFAULT_Q_VALUE_CUTOFF
        )

        # Final MBR identification: for the two-pass (transfer learning)
        # search, a precursor-run ID is only accepted when it passes *both*
        # the library-level cutoff (first-pass global q-value, carried via
        # library_confidence.parquet) and the second-pass run-specific
        # cutoff. The second pass's own global_precursor_q_value is
        # diagnostic only and is not used here.
        # For a single-pass search (or the first-pass-only snapshot) the
        # existing (looser) OR logic is kept.
        if two_pass_mode:
            pmsm_df = pmsm_df.filter(
                (pl.col(library_q_value_column) <= q_value_cutoff)
                & (pl.col("precursor_q_value") <= q_value_cutoff)
            )
        else:
            pmsm_df = pmsm_df.filter(
                (pl.col("precursor_q_value") <= q_value_cutoff)
                | (pl.col(library_q_value_column) <= q_value_cutoff)
            )

        ## Add modified sequence column
        pmsm_df = self._add_report_columns(pmsm_df)

        if not output_decoy:
            pmsm_df = pmsm_df.filter(pl.col("is_decoy") == False)

        if format == "parquet":
            pmsm_df.write_parquet(
                self.search_config.output_dir / f"{filename_stem}.parquet"
            )
            if pg_quant_df is not None:
                pg_quant_df.write_parquet(
                    self.search_config.output_dir
                    / "protein_group_maxlfq_results.parquet"
                )
        else:
            pmsm_df.with_columns(
                pl.col("protein_index").cast(pl.List(pl.String)).list.join(";")
            ).write_csv(
                self.search_config.output_dir / f"{filename_stem}.tsv", separator="\t"
            )
            if pg_quant_df is not None:
                pg_quant_df.write_csv(
                    self.search_config.output_dir / "protein_group_maxlfq_results.tsv",
                    separator="\t",
                )

    def check_result_files(self) -> None:
        """Raise FileExistsError if any per-run result file already exists."""
        existing_files = [
            ResultManager.get_hdf_file_path(self.output_dir, run_name)
            for run_name in self.search_config.run_names
            if ResultManager.get_hdf_file_path(self.output_dir, run_name).exists()
        ]
        if existing_files:
            existing_names = ", ".join(f.name for f in existing_files)
            raise FileExistsError(
                f"Result file(s) already exist in '{self.output_dir}': {existing_names}. "
                "Remove them before starting a new search."
            )

    def execute_workflow(self) -> None:
        """
        Execute the full search workflow.
        """
        logger.info(
            f"DelPi workflow started with configuration: {self.search_config.yaml_path}"
        )

        logger.info(
            f"Search configuration:\n\n{yaml.dump(self.search_config.config, default_flow_style=False, indent=2)}"
        )

        self.check_result_files()

        enable_tl = self.search_config.enable_transfer_learning
        if enable_tl:
            logger.info("Two-stage search (transfer learning enabled)")
        else:
            logger.info("Single-stage search (transfer learning disabled)")

        self.prepare_database()

        # First pass: search every run
        self.execute_batch()

        # First pass: ONE global (cross-run) re-scoring after all runs are
        # searched, computing both run-specific and global q-values. Protein
        # grouping is only performed here (first pass).
        first_pmsm_df = self.perform_global_tda(
            SearchState.FIRST_TDA, run_protein_grouping=True
        )

        if enable_tl:
            self.save_pmsm_df(first_pmsm_df, filename_stem="pmsm_results.first")

            # Transfer learning: dual-FDR + top-k target selection for
            # predictor fine-tuning, refined target-decoy library
            # construction (paired decoys) and library confidence.
            rt_predictor, ms2_predictor = self.perform_transfer_learning(first_pmsm_df)

            self.build_refined_library(first_pmsm_df, rt_predictor, ms2_predictor)

            # Second pass: re-search every run against the refined library
            self.execute_batch()

            # Second pass: global re-scoring against the refined library.
            # Target protein-group membership + library q-values are reused
            # from the first pass and decoys are freshly grouped (see
            # FDRAnalyzer.perform_global_analysis); the resulting global
            # q-value here is diagnostic only, so results are reported/
            # filtered using the first-pass-derived library q-value instead.
            pmsm_df = self.perform_global_tda(
                SearchState.SECOND_TDA, run_protein_grouping=False
            )
            library_q_value_column = "library_precursor_q_value"
        else:
            pmsm_df = first_pmsm_df
            library_q_value_column = "global_precursor_q_value"

        # Quantification
        pmsm_df, pg_quant_df = self.perform_quantification(
            pmsm_df, library_q_value_column=library_q_value_column
        )

        # Save final results
        self.save_pmsm_df(
            pmsm_df,
            pg_quant_df,
            library_q_value_column=library_q_value_column,
            two_pass_mode=enable_tl,
        )
        self.state = SearchState.DONE

        logger.info("DelPi workflow completed successfully")

    def log_id_statistics_table(
        self, pmsm_df, q_value_cutoff, use_library_q_value: bool = False
    ):
        """Log a summary table of global and per-run identification counts.

        ``use_library_q_value`` is set for the second pass of the two-pass
        MBR search: the first-pass-derived library q-value (a fixed
        per-precursor value, not run-specific) is used for both the global
        and every per-run row instead of this pass's own (diagnostic-only)
        global/run-specific q-value columns.
        """

        search_config = self.search_config
        rows = []

        # Global summary
        counts = ResultManager.compute_id_statistics(
            pmsm_df,
            q_value_cutoff,
            global_fdr=True,
            use_library_q_value=use_library_q_value,
        )
        rows.append(
            [
                "Global",
                counts["precursors"],
                counts["peptides"],
                counts["protein_groups"],
            ]
        )

        # Per-run summaries
        for run_index, run_name in enumerate(search_config.run_names):
            counts = ResultManager.compute_id_statistics(
                pmsm_df.filter(pl.col("run_index") == run_index),
                q_value_cutoff,
                use_library_q_value=use_library_q_value,
            )
            rows.append(
                [
                    run_name,
                    counts["precursors"],
                    counts["peptides"],
                    counts["protein_groups"],
                ]
            )

        table = tabulate(
            rows,
            headers=["Scope", "Precursors", "Peptides", "Protein Groups"],
            tablefmt="psql",  # or "psql", "fancy_grid", "grid"
            numalign="right",
            stralign="left",
        )

        logger.info(f"Identification summary @ {q_value_cutoff:.2f} FDR\n" + table)
