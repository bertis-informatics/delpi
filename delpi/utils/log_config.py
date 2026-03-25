import logging
import warnings
import sys
from pathlib import Path

from tqdm import tqdm
from lightning.pytorch.utilities.warnings import PossibleUserWarning

# LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
LOG_FORMAT = "[%(asctime)s][%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%m-%d-%Y %H:%M:%S"


class TqdmLoggingHandler(logging.StreamHandler):
    """StreamHandler that writes via ``tqdm.write`` to avoid corrupting
    active tqdm progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def configure_logging(
    logfile_path: str = None, level: int = logging.INFO, use_tqdm: bool = False
):

    handlers = []
    log_formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # 1. stdout handler (tqdm-compatible when tqdm bars are active)
    if use_tqdm:
        stream_handler = TqdmLoggingHandler(sys.stdout)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    handlers.append(stream_handler)

    # 2. file handler
    if logfile_path:
        logfile_path = Path(logfile_path)
        if not logfile_path.parent.exists():
            raise FileNotFoundError(f"{logfile_path.parent} doesn't exist")
        file_handler = logging.FileHandler(logfile_path, mode="a")
        file_handler.setFormatter(log_formatter)
        handlers.append(file_handler)

    # 3. basic config
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # 4. ignore false positive warnings from lightning
    warnings.filterwarnings("ignore", category=PossibleUserWarning)
