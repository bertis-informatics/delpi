import logging
import warnings
import sys
from pathlib import Path

from lightning.pytorch.utilities.warnings import PossibleUserWarning

# LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
LOG_FORMAT = "[%(asctime)s][%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%m-%d-%Y %H:%M:%S"


def configure_logging(logfile_path: str = None, level: int = logging.INFO):

    handlers = []
    log_formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # 1. stdout handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    handlers.append(stream_handler)

    # 2. file handler
    if logfile_path:
        logfile_path = Path(logfile_path)
        if not logfile_path.parent.exists():
            raise FileNotFoundError(f"{logfile_path.parent} doesn't exist")
        # logfile_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logfile_path, mode="a")
        file_handler.setFormatter(log_formatter)
        handlers.append(file_handler)

    # 3. basic config
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # 4. suppress bmsio logs
    logging.getLogger("bmsio").setLevel(logging.ERROR)

    # 5. ignore false positive warnings from lightning
    warnings.filterwarnings("ignore", category=PossibleUserWarning)
