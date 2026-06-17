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
from delpi.search.tda.tda_processor import TDAProcessor
from delpi.search.search_state import SearchState
from delpi.search.progress import CallbackProgressTracker
from delpi.search.dia.max_lfq import maxlfq
from delpi.utils.mp import get_multiprocessing_context
from delpi.database.utils import get_modified_sequence
from delpi.constants import DEFAULT_Q_VALUE_CUTOFF

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

    def perform_transfer_learning(self) -> None:

        self.state = SearchState.TL_TRAINING
        output_dir = self.search_config.output_dir
        search_config = self.search_config

        logger.info("Transfer learning started")

        result_aggregator = ResultsAggregator(
            db_dir=self.get_db_dir(), search_config=search_config
        )
        logger.info("Training RT predictor")
        rt_trainer = TransferLearningTrainerForRT()
        rt_predictor = rt_trainer.train(
            output_dir=output_dir,
            result_aggregator=result_aggregator,
            device=self.device,
        )
        del rt_trainer

        logger.info("Training MS2 spectrum predictor")
        trainer = TransferLearningTrainer()
        ms2_predictor = trainer.train(
            output_dir=output_dir,
            result_aggregator=result_aggregator,
            device=self.device,
        )
        del trainer

        self.state = SearchState.REFINED_DB_PREP
        precursor_index_arr = []

        logger.info("Generating refined spectral library with fine-tuned models")
        for run_idx, ret_mgr in result_aggregator._results_dict.items():
            results_dict = ret_mgr.read_dict(
                "first_results", data_keys=["precursor_index"]
            )
            precursor_index_arr.append(results_dict["precursor_index"])

        precursor_index_arr = np.unique(np.concatenate(precursor_index_arr))
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
            precursor_index_arr,
        )
        spec_generator.save(search_config.refined_db_dir)

    def perform_global_tda(self) -> None:
        self.state = SearchState.SECOND_TDA
        logger.info("Performing global target-decoy analysis")

        search_config = self.search_config
        group_key = self.get_results_group_key()
        q_value_cutoff = search_config.config.get(
            "q_value_cutoff", DEFAULT_Q_VALUE_CUTOFF
        )
        use_protein_picker = search_config.config.get("use_protein_picker", True)

        result_aggregator = ResultsAggregator(
            db_dir=self.get_db_dir(), search_config=search_config
        )

        search_batch_size = search_config.config.get("batch_size", 512)
        processor = TDAProcessor(
            db_dir=self.get_db_dir(),
            output_dir=search_config.output_dir,
            device=self.device,
            q_value_cutoff=q_value_cutoff,
            use_protein_picker=use_protein_picker,
            batch_size=search_batch_size * 4,
            split_level="peptide",
        )
        pmsm_df = processor.run_global(
            result_aggregator,
            group_key,
            training_params={
                "num_warmup_steps": 5,
                "max_epochs": 50,
                "train_split": 0.8,
                "early_stopping_patience": 5,
            },
        )
        self.log_id_statistics_table(pmsm_df, q_value_cutoff)
        return pmsm_df

    def perform_quantification(self, pmsm_df: pl.DataFrame) -> None:

        logger.info("Performing cross-run quantification")
        self.state = SearchState.QUANTIFICATION
        q_value_cutoff = self.search_config.config.get(
            "q_value_cutoff", DEFAULT_Q_VALUE_CUTOFF
        )

        result_aggregator = ResultsAggregator(
            db_dir=self.get_db_dir(), search_config=self.search_config
        )

        lfq = LabelFreeQuantifier(
            result_aggregator,
            q_value_cutoff=q_value_cutoff,
            group_key=self.get_results_group_key(),
            acq_method=self.search_config.config.get("acquisition_method", "DDA"),
        )

        quant_df = lfq.perform_quantification(pmsm_df)
        pmsm_df = (
            pmsm_df.select(pl.exclude("ms1_area", "ms2_area"))
            .group_by(["run_index", "precursor_index"])
            .agg(pl.all().sort_by("score").last())
            .join(quant_df, on=["run_index", "precursor_index"], how="left")
        )

        ## run MaxLFQ
        if self.search_config.config.get("acquisition_method", "DDA").upper() == "DIA":
            logger.info("Performing protein quantification with MaxLFQ ")
            df = (
                pmsm_df.filter(pl.col("is_decoy") == False)
                .filter(pl.col("global_protein_group_q_value") <= q_value_cutoff)
                .filter(pl.col("ms2_area").is_not_null() & (pl.col("ms2_area") > 0))
            )
            pg_quant_df = maxlfq(
                df,
                min_peptides_per_protein=1,
                peptide_col="precursor_index",
                intensity_col="ms2_area",
            )
            pg_quant_df = pg_quant_df.join(
                result_aggregator.get_run_df(), on="run_index", how="left"
            )
        else:
            pg_quant_df = None

        return pmsm_df, pg_quant_df

    def save_pmsm_df(self, pmsm_df: pl.DataFrame, pg_quant_df: pl.DataFrame) -> None:

        format = self.search_config.config.get("output_format", "tsv").lower()
        output_decoy = self.search_config.config.get("output_decoy", True)
        q_value_cutoff = self.search_config.config.get(
            "q_value_cutoff", DEFAULT_Q_VALUE_CUTOFF
        )

        # filter by q-value and add run information
        pmsm_df = pmsm_df.filter(
            (pl.col("precursor_q_value") <= q_value_cutoff)
            | (pl.col("global_precursor_q_value") <= q_value_cutoff)
        )

        ## Add modified sequence column
        pmsm_df = pmsm_df.with_columns(
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

        if not output_decoy:
            pmsm_df = pmsm_df.filter(pl.col("is_decoy") == False)

        if format == "parquet":
            pmsm_df.write_parquet(
                self.search_config.output_dir / "pmsm_results.parquet"
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
                self.search_config.output_dir / "pmsm_results.tsv", separator="\t"
            )
            if pg_quant_df is not None:
                pg_quant_df.write_csv(
                    self.search_config.output_dir / "protein_group_maxlfq_results.tsv",
                    separator="\t",
                )

        self.state = SearchState.DONE

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

        # First search
        self.execute_batch()

        if enable_tl:
            # Transfer learning
            self.perform_transfer_learning()

            # Second search
            self.execute_batch()

        # FDR control and quantification
        pmsm_df = self.perform_global_tda()

        # pmsm_df.write_parquet(self.search_config.output_dir / "pmsm_scores.parquet")

        # Quantification
        pmsm_df, pg_quant_df = self.perform_quantification(pmsm_df)

        # Save final results
        self.save_pmsm_df(pmsm_df, pg_quant_df)

        logger.info("DelPi workflow completed successfully")

    def log_id_statistics_table(self, pmsm_df, q_value_cutoff):

        search_config = self.search_config
        rows = []

        # Global summary
        counts = ResultManager.compute_id_statistics(
            pmsm_df, q_value_cutoff, global_fdr=True
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
                pmsm_df.filter(pl.col("run_index") == run_index), q_value_cutoff
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
