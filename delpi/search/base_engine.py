"""Base search engine shared by the DDA and DIA pipelines."""

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
from delpi.model.pmsm_scale import PeptideMultiSpectraMatchScaler
from delpi.model.rt_calibrator import (
    RetentionTimeCalibrator,
    LinearProjectionCalibrator,
)
from delpi.search.tl.data_prep import TransferLearningDataPreparator
from delpi.search.search_state import SearchState
from delpi.search.progress.tracker import ProgressTracker
from delpi.search.progress.tqdm_tracker import TqdmProgressTracker
from delpi.search.progress.callback_tracker import CallbackProgressTracker
from delpi.utils.fdr import calculate_q_value
from delpi.utils.log_config import configure_logging
from delpi import MODEL_DIR

logger = logging.getLogger(__name__)


class BaseSearchEngine(ABC):
    """Abstract base for the DDA and DIA search engines.

    Holds the search config, device and search state, runs one raw file per
    child process and does the RT calibration shared by both engines.
    """

    def __init__(
        self,
        search_config: SearchConfig,
        device: torch.device,
        state: SearchState = SearchState.INIT,
    ):
        self.search_config = search_config
        self.device = device
        self.state = state

    def _load_model(self) -> DelPiModel:
        model_path = MODEL_DIR / f"delpi.unified.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = DelPiModel.load(model_path).to(self.device)
        model.transform = PeptideMultiSpectraMatchScaler()
        return model

    def get_min_search_fragments(self, speclib_reader) -> int:
        """Configured minimum fragment count, checked against the library cap."""
        value = self.search_config.min_search_fragments
        if value > speclib_reader.max_fragments:
            raise ValueError(
                f"fragment.min_fragments ({value}) exceeds the library's "
                f"max_fragments ({speclib_reader.max_fragments}); no precursor "
                "would be searched"
            )
        return value

    @staticmethod
    def _limit_numba_threads():
        """Use half of the numba threads in the current (background) thread.

        Passed as ``thread_initializer`` to :class:`Prefetcher`, so background
        ``find_peak_groups`` work leaves cores free for the GPU data pipeline.
        """
        import numba

        n = max(1, numba.config.NUMBA_NUM_THREADS // 2)
        numba.set_num_threads(n)

    def get_save_quant(self) -> bool:
        return (
            self.state >= SearchState.SECOND_SEARCH or not self.search_config.enable_mbr
        )

    def get_logit_cutoff(self, base_cutoff: float) -> float:
        return (
            base_cutoff - 4.0
            if self.state >= SearchState.SECOND_SEARCH
            else base_cutoff
        )

    def next_state(self):
        self.state = SearchState(self.state + 1)

    def get_db_dir(self):
        if not self.search_config.enable_mbr:
            return self.search_config.db_dir
        return (
            self.search_config.db_dir
            if self.state < SearchState.SECOND_SEARCH
            else self.search_config.refined_db_dir
        )

    def get_results_group_key(self):
        if not self.search_config.enable_mbr:
            return "first_results"
        return (
            "first_results"
            if self.state < SearchState.SECOND_SEARCH
            else "second_results"
        )

    @abstractmethod
    def perform_quick_search(self, run: Union[DIARun, DDARun]) -> pl.DataFrame:
        pass

    def perform_rt_bootstrap(
        self, run: Union[DIARun, DDARun], figure_path: Path = None
    ):
        """Hook for the initial DIA RT bootstrap (delpi.search.dia.rt_bootstrap).

        `_perform_rt_calibration` calls it before the first full search and
        uses the result without a fallback, so engines that take that path
        (DIA) must override it. DDA never does and keeps this None default.
        """
        return None

    @abstractmethod
    def perform_search(
        self, lcms_data: MassSpecData, progress: ProgressTracker = None
    ) -> ResultManager:
        """Run the acquisition-specific search on one LC-MS/MS run."""
        pass

    @abstractmethod
    def get_acquisition_method(self) -> str:
        """Return "DDA" or "DIA"."""
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
        """Run the full search pipeline for one raw file.

        Called in a child process spawned by :meth:`SearchManager.execute_batch`.

        Args:
            progress_queue: if given, :class:`ProgressSnapshot` objects are put
                on it for the parent, then a ``None`` sentinel when the method
                finishes (success or error). Otherwise progress goes to tqdm.
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
                progress_queue.put(None)  # sentinel: the run is done

    def prepare_tl_data_for_run(
        self,
        raw_path: Path,
        target_pmsm_df: pl.DataFrame,
        tl_ms2_h5_path: Path,
        progress_queue=None,
    ) -> None:
        """Extract MS2/RT transfer-learning data for one run into the shared
        ``transfer_learning_ms2.h5``.

        `target_pmsm_df` holds this run's target PmSMs after the first-pass
        global target-decoy analysis, already dual-FDR filtered and top-k selected
        (:func:`delpi.search.tl.second_pass.select_tl_training_pmsms`). Called
        in a child process spawned by
        :meth:`SearchManager.prepare_transfer_learning_training_data`, like
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
                nce=self.search_config.nce,
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
                progress_queue.put(None)  # sentinel: the run is done

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

        # before a full search, use the first-search results
        if before_full_search:
            group_key = "first_results"
            # current DB, not the original: the refined DB's ref_rt comes from
            # the fine-tuned RT model
            # db_dir = self.search_config.db_dir
            db_dir = self.get_db_dir()
        else:
            group_key = self.get_results_group_key()
            db_dir = self.get_db_dir()

        if self.state == SearchState.FIRST_SEARCH and before_full_search:
            bootstrap_result = self.perform_rt_bootstrap(run, figure_path=fig_path)
            if not bootstrap_result.success:
                logger.warning(
                    f"DIA RT bootstrap calibration fell back to "
                    f"broad bounds (reason={bootstrap_result.fallback_reason}, "
                    f"rounds={bootstrap_result.n_rounds}, "
                    f"evaluated={bootstrap_result.n_evaluated}, "
                    f"anchors={bootstrap_result.n_unique_anchors})"
                )
            return bootstrap_result.calibrator
            # legacy path, before the RT bootstrap:
            # q_value_cutoff = 0.05
            # pmsm_df = self.perform_quick_search(run)
            # target_df = pmsm_df.filter(pl.col("is_decoy") == False)
        else:
            q_value_cutoff = 0.01
            if after_full_search:
                results_dict = result_manager.read_dict(
                    group_key, data_keys=["precursor_index", "observed_rt", "logit"]
                )
                # row index of the stored results: joins below need not keep row order
                pmsm_df = pl.DataFrame(results_dict).with_row_index("pmsm_index")
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
            else:  # SECOND_SEARCH, before the full search
                results_dict = result_manager.read_dict(
                    group_key,
                    data_keys=[
                        "precursor_index",
                        "observed_rt",
                        "predicted_rt",
                        "score",
                    ],
                )
                pmsm_df = (
                    pl.DataFrame(results_dict)
                    .group_by("precursor_index")
                    .agg(pl.all().sort_by("score").last())
                )

                # map precursor_index to the refined DB
                precursor_df = (
                    pl.scan_parquet(db_dir / "precursor_df.parquet")
                    .select(pl.col("g_precursor_index", "precursor_index"))
                    .collect()
                )
                pmsm_df = pmsm_df.rename({"precursor_index": "g_precursor_index"}).join(
                    precursor_df.select(pl.col("g_precursor_index", "precursor_index")),
                    on="g_precursor_index",
                    how="inner",  # consider only library-filtered precursors
                )

                pmsm_df = PeptideDatabase.join(
                    db_dir,
                    pmsm_df,
                    precursor_columns=[],
                    modification_columns=["ref_rt"],
                    peptide_columns=["is_decoy"],
                )
                pmsm_df = calculate_q_value(pmsm_df, out_column="precursor_q_value")

            target_df = pmsm_df.filter(
                (pl.col("precursor_q_value") <= q_value_cutoff)
                & (pl.col("is_decoy") == False)
            )

        if target_df.height < 150:
            rt_calibrator = LinearProjectionCalibrator(
                min_rt_in_seconds=meta_df.item(0, "time_in_seconds"),
                max_rt_in_seconds=meta_df.item(-1, "time_in_seconds"),
                rt_tolerance=0.25,
            )
        else:
            rt_calibrator = RetentionTimeCalibrator.train(
                min_rt_in_seconds=meta_df.item(0, "time_in_seconds"),
                max_rt_in_seconds=meta_df.item(-1, "time_in_seconds"),
                ref_rt=target_df["ref_rt"].to_numpy(),
                obs_rt=target_df["observed_rt"].to_numpy(),
                degree=2,
                min_rt_tolerance=0.1,
                max_rt_tolerance=0.11,
            )

        if after_full_search:
            # save RT predictions for the full search results, in stored row order
            pmsm_df = pmsm_df.sort("pmsm_index")
            rt_df = rt_calibrator.predict(pmsm_df["ref_rt"])
            result_manager.write_dict(
                group_key, {"predicted_rt": rt_df["predicted_rt"].to_numpy()}
            )

        return rt_calibrator
