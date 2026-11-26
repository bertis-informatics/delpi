import logging
import argparse
import sys
from pathlib import Path
from typing import Union

from delpi.search.config import SearchConfig
from delpi.search.search_manager import SearchManager
from delpi.utils.log_config import configure_logging

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
    """Run a DelPi search from a configuration YAML file.

    Args:
        batch_size: inference batch size, or 'auto' to set it from GPU memory
        progress: optional CallbackProgressTracker; progress from the search
            child processes is forwarded to its callback (e.g. for a GUI).
            None shows a tqdm bar instead.
    """

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    search_config = SearchConfig(config_path)
    search_config.config["batch_size"] = batch_size
    search_config.check_params()

    log_level_num = getattr(logging, log_level.upper())
    configure_logging(logfile_path=search_config.log_file_path, level=log_level_num)

    search_mgr = SearchManager(
        search_config, specified_device=device, progress=progress
    )
    search_mgr.execute_workflow()


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
