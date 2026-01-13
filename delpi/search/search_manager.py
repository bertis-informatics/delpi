"""
Search Coordinator for DelPi

This module provides the main search workflow coordination and factory for creating
acquisition-specific search engines.
"""

import logging
import time
import yaml
import psutil
from pathlib import Path

import polars as pl
import numpy as np
import torch
from lightning.pytorch.accelerators import AcceleratorRegistry
from tabulate import tabulate

from delpi.lcms.reader.base import MassSpecFileReader
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
from delpi.search.tda.dataset import DatasetSplitter, PMSMDataset
from delpi.search.tda.fdr_analyzer import FDRAnalyzer
from delpi.search.tda.trainer import TargetDecoyTrainer
from delpi.search.search_state import SearchState
from delpi.search.dia.max_lfq import maxlfq
from delpi.search.rt_align import (
    align_retention_times,
    estimate_representative_retention_time,
)
from delpi.utils.mp import get_multiprocessing_context
from delpi.database.utils import get_modified_sequence


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

    def __init__(self, search_config: SearchConfig, specified_device: str = "auto"):
        """
        Initialize the search coordinator.

        Args:
            search_config: Configuration object containing search parameters
        """
        self.search_config: SearchConfig = search_config
        self._validated_device: torch.device = None
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
            from delpi.search.database import build_database_in_subprocess

            build_database_in_subprocess(self.search_config, self.device)
        else:
            logger.info("Peptide database found (previously built)")

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

    def execute_batch(self) -> None:
        """Execute workflow for all input files using separate processes."""

        input_files = self.input_files

        if self.state < SearchState.FIRST_SEARCH:
            self.state = SearchState.FIRST_SEARCH
        else:
            self.state = SearchState.SECOND_SEARCH
            logger.info(f"Second search after transfer learning")

        logger.info(f"Total runs to process: {len(input_files)}")

        # Create engine instance for process execution
        engine = self.get_engine()

        for run_idx, raw_path in enumerate(input_files):
            run_name = MassSpecFileReader.extract_run_name(raw_path)
            st_t = time.perf_counter()
            logger.info(f"[{run_idx+1}/{len(input_files)}] Processing run: {run_name}")

            p = get_multiprocessing_context().Process(
                target=engine.process_single_run,
                args=(raw_path,),
            )
            p.start()
            logger.debug(f"Start a child process (PID: {p.pid})")
            p.join()

            if p.exitcode != 0:
                raise RuntimeError(f"Searching failed with exit code {p.exitcode}")
            logger.debug(f"Terminate child process (PID: {p.pid})")

            elapsed = time.perf_counter() - st_t
            logger.info(
                f"[{run_idx+1}/{len(input_files)}] Completed processing. Elapsed: {elapsed:.1f} s"
            )

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
            ms2_predictor=ms2_predictor,
            rt_predictor=rt_predictor,
        )

        spec_generator.generate_spectral_lib(search_config.db_dir, precursor_index_arr)
        spec_generator.save(search_config.refined_db_dir)

    def perform_global_tda(self) -> None:

        # [TODO] handle potential memory issue in case of many runs

        self.state = SearchState.SECOND_TDA
        logger.info("Performing global target-decoy analysis")

        search_config = self.search_config
        is_dia = search_config.config.get("acquisition_method", "DIA") == "DIA"
        pmsm_sel_col = "apex_median_intensity" if is_dia else "score"

        group_key = self.get_results_group_key()
        q_value_cutoff = search_config.config["q_value_cutoff"]

        result_aggregator = ResultsAggregator(
            db_dir=self.get_db_dir(), search_config=search_config
        )

        hdf_files = result_aggregator.get_hdf_files()
        num_runs = len(result_aggregator._results_dict)

        # [TODO] load features conditionally based on num_runs and available memory
        total_hdf_size = sum(f.stat().st_size for f in hdf_files)
        available_memory = psutil.virtual_memory().available
        load_features = total_hdf_size * 0.4 < available_memory
        # load_features = num_runs <= 64
        logger.info(
            f"Total HDF size: {total_hdf_size / (1024**3):.2f} GB, "
            f"Available RAM: {available_memory / (1024**3):.2f} GB. "
            f"Load features: {load_features}"
        )

        pmsm_df, data_dict = result_aggregator.get_search_results(
            group_key, load_features=load_features, estimate_apex_intensity=is_dia
        )

        num_decoys = pmsm_df["is_decoy"].sum()
        num_targets = len(pmsm_df) - num_decoys
        logger.info(
            f"Training a classifier with {num_targets:,} positive and {num_decoys:,} negative PMSMs"
        )

        # Train classifier and rescore PMSMs
        splitter = DatasetSplitter()
        train_df, test_df = splitter.split_by_peptide(pmsm_df)

        if load_features:
            train_dataset = PMSMDataset.create_tensor_dataset(train_df, data_dict)
            test_dataset = PMSMDataset.create_tensor_dataset(test_df, data_dict)
        else:
            train_dataset = PMSMDataset(
                pmsm_df=train_df,
                hdf_files=hdf_files,
                hdf_group_key=group_key,
            )
            test_dataset = PMSMDataset(
                pmsm_df=test_df,
                hdf_files=hdf_files,
                hdf_group_key=group_key,
            )

        trainer = TargetDecoyTrainer()
        test_score_arr = trainer.train(
            model_version="global_tda",
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_dir=search_config.output_dir,
            device=self.device,
        )

        # Cleanup (make sure hdf files are closed)
        del test_dataset
        del train_dataset

        ## select best-scoring PMSM per cluster of each run
        pmsm_df = (
            test_df.with_columns(score=test_score_arr)
            .group_by(["run_index", "cluster"])
            .agg(pl.all().sort_by("score").last())
        )
        logger.info(
            f"Selected {pmsm_df.shape[0]} non-redundant PMSMs (one per cluster) from {test_df.shape[0]} PMSMs"
        )

        ## align retention times and estimate representative RTs
        logger.info("Aligning retention times across runs")
        xic_peak_interval = result_aggregator.get_xic_peak_interval()

        aligned_rt_df = align_retention_times(
            pmsm_df,
            rt_column="observed_rt",
            aligned_rt_column="aligned_rt",
            pmsm_sel_col=pmsm_sel_col,
        )
        rep_rt_df = estimate_representative_retention_time(
            aligned_rt_df,
            weight_column="score",
            aligned_rt_column="aligned_rt",
            xic_peak_interval=xic_peak_interval,
            rep_rt_column="representative_rt",
        )

        ## select PMSM for each precursor based on aligned RT
        logger.info("Selecting best PMSM per precursor based on aligned RT")
        precursor_rt_df = (
            aligned_rt_df.join(rep_rt_df, on="precursor_index", how="left")
            .with_columns(
                (pl.col("aligned_rt") - pl.col("representative_rt"))
                .abs()
                .alias("delta_rt")
            )
            .group_by(["run_index", "precursor_index"])
            .agg(pl.all().sort_by("delta_rt").first())
        )
        pmsm_df = pmsm_df.join(
            precursor_rt_df.select(pl.col("run_index", "pmsm_index")),
            on=["run_index", "pmsm_index"],
            how="inner",
        )

        ## update precursor score with the best PMSM score
        # precursor_score_df = pmsm_df.group_by(["run_index", "precursor_index"]).agg(
        #     pl.col("score").max()
        # )
        # pmsm_df = (
        #     pmsm_df.select(pl.exclude("score"))
        #     .join(
        #         precursor_rt_df.select(pl.col("run_index", "pmsm_index")),
        #         on=["run_index", "pmsm_index"],
        #         how="inner",
        #     )
        #     .join(precursor_score_df, on=["run_index", "precursor_index"], how="left")
        # )
        logger.debug(
            f"Selected {pmsm_df.shape[0]} best-scoring PMSMs (one per Precursor)"
        )

        fdr_analyzer = FDRAnalyzer(
            q_value_cutoff=q_value_cutoff, db_dir=search_config.db_dir
        )
        pmsm_df = fdr_analyzer.perform_global_analysis(pmsm_df)
        pmsm_df = fdr_analyzer.batch_run_specific_analysis(pmsm_df)
        pmsm_df.write_parquet(self.search_config.output_dir / "pmsm.bak.parquet")
        # pmsm_df = pl.read_parquet(self.search_config.output_dir / "pmsm.bak.parquet")

        pmsm_df = pmsm_df.filter(
            (pl.col("precursor_q_value") <= q_value_cutoff)
            | (pl.col("global_precursor_q_value") <= q_value_cutoff)
        ).join(result_aggregator.get_run_df(), on="run_index", how="left")
        # for hdf_file_path in result_aggregator.get_hdf_files():
        #     with h5py.File(hdf_file_path, mode="a") as hdf_file:
        #         if "filtered_results" in hdf_file:
        #             del hdf_file["filtered_results"]

        result_aggregator.save_filtered_results(
            pmsm_df, src_group_key=group_key, dst_group_key="filtered_results"
        )
        self.log_id_statistics_table(pmsm_df, q_value_cutoff)

        return pmsm_df

    def perform_quantification(self, pmsm_df: pl.DataFrame) -> None:

        logger.info("Performing cross-run quantification")
        self.state = SearchState.QUANTIFICATION

        result_aggregator = ResultsAggregator(
            db_dir=self.get_db_dir(), search_config=self.search_config
        )

        lfq = LabelFreeQuantifier(
            result_aggregator,
            group_key="filtered_results",
            acq_method=self.search_config.config.get("acquisition_method", "DDA"),
            select_topk_fragments=6,
        )
        quant_df = lfq.perform_quantification()
        pmsm_df = pmsm_df.select(pl.exclude("ms1_area", "ms2_area")).join(
            quant_df, on=["run_index", "precursor_index"], how="left"
        )

        ## run MaxLFQ
        if self.search_config.config.get("acquisition_method", "DDA").upper() == "DIA":
            df = (
                pmsm_df.filter(pl.col("is_decoy") == False)
                .filter(pl.col("global_protein_group_q_value") <= 0.01)
                .filter(pl.col("ms2_area").is_not_null() & (pl.col("ms2_area") > 0))
                .group_by(["protein_group", "peptide_index", "run_name"])
                .agg(
                    pl.col("ms2_area").median().alias("peptide_abundance"),
                )
            )
            pg_quant_df = maxlfq(df, run_col="run_name", min_peptides_per_protein=1)
        else:
            pg_quant_df = None

        return pmsm_df, pg_quant_df

    def save_pmsm_df(
        self, pmsm_df: pl.DataFrame, pg_quant_df: pl.DataFrame, target_only: bool = True
    ) -> None:

        format = self.search_config.config.get("report_format", "tsv").lower()

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

        if target_only:
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

        self.prepare_database()

        # First search
        self.execute_batch()

        # Transfer learning
        self.perform_transfer_learning()

        # Second search
        self.execute_batch()

        # FDR control and quantification
        pmsm_df = self.perform_global_tda()

        # result_aggregator = ResultsAggregator(
        #     db_dir=self.get_db_dir(), search_config=self.search_config
        # )
        # pmsm_df = (
        #     pl.read_parquet(self.search_config.output_dir / "pmsm.bak.parquet")
        #     .filter(
        #         (pl.col("precursor_q_value") <= 0.01)
        #         | (pl.col("global_precursor_q_value") <= 0.01)
        #     )
        #     .join(result_aggregator.get_run_df(), on="run_index", how="left")
        # )

        pmsm_df, pg_quant_df = self.perform_quantification(pmsm_df)

        # Save final results
        self.save_pmsm_df(pmsm_df, pg_quant_df, target_only=True)

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
