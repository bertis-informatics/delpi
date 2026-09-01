import logging
import argparse
import sys
from pathlib import Path
from typing import Union
import polars as pl

from delpi.search.config import SearchConfig
from delpi.search.search_manager import SearchManager
from delpi.search.search_state import SearchState
from delpi.utils.log_config import configure_logging
import yaml

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DelPi: Deep Learning-based Peptide Identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_path", type=str, help="Path to the configuration YAML file"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (e.g., 'cuda', 'cuda:0', 'cuda:1')",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size for search inference ('auto' to set based on GPU memory)",
    )

    return parser.parse_args()


def run_search(
    config_path: str,
    device: str = "cuda:0",
    log_level: str = "info",
    batch_size: Union[str, int] = "auto",
    progress=None,
):
    # config_path = r"/data1/benchmark/DIA/2025-SCP/delpi-test/params.yaml"
    # device: str = "cuda:0"
    # log_level: str = "info"
    # batch_size = 512
    # progress = None

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Initialize search configuration
    search_config = SearchConfig(config_path)
    search_config.config["batch_size"] = batch_size
    search_config.check_params()

    # Configure logging
    log_level_num = getattr(logging, log_level.upper())
    configure_logging(logfile_path=search_config.log_file_path, level=log_level_num)

    # Run search
    search_mgr = SearchManager(
        search_config, specified_device=device, progress=progress
    )
    self = search_mgr
    # search_mgr.execute_workflow()

    # self.check_result_files()
    enable_tl = self.search_config.enable_transfer_learning
    if enable_tl:
        logger.info("Two-stage search (transfer learning enabled)")
    else:
        logger.info("Single-stage search (transfer learning disabled)")

    self.prepare_database()

    # self.execute_batch()

    # first_pmsm_df = self.perform_global_tda(
    #     SearchState.FIRST_TDA, run_protein_grouping=True
    # )

    if enable_tl:
        # self.save_pmsm_df(first_pmsm_df, filename_stem="pmsm_results.first")

        # first_pmsm_df = pl.read_parquet(self.output_dir / "pmsm_results.first.parquet")

        # rt_predictor, ms2_predictor = self.perform_transfer_learning(first_pmsm_df)

        # self.build_refined_library(first_pmsm_df, rt_predictor, ms2_predictor)

        # Second pass: re-search every run against the refined library
        # self.execute_batch()

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
        first_pmsm_df = pl.read_parquet(self.output_dir / "pmsm_results.first.parquet")
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


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        run_search(
            config_path=args.config_path,
            device=args.device,
            log_level=args.log_level,
            batch_size=args.batch_size,
        )
    except Exception as e:
        logger.error(f"DelPi search failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
