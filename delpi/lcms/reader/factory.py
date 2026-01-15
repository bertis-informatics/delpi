from pathlib import Path
from typing import Union

from .mzml import MzmlFileReader
from .hdf import HdfReader
from .base import MassSpecFileReader

try:
    from .bmsio import BMSIOReader
except ImportError:
    BMSIOReader = None

from pymsio.readers.thermo import ThermoRawReader

class ReaderFactory:

    supported_file_extensions = ["raw", "mzml", "mzml.gz", "hdf", "h5"]

    @classmethod
    def get_reader(
        cls,
        filepath: Union[Path, str],
        run_name: str = None,
        # bmsio_server_addr: str = "localhost:5001",
        acquisition_method: str = None,
    ) -> MassSpecFileReader:

        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Handle .mzml.gz files
        if str(filepath).lower().endswith(".mzml.gz"):
            file_extension = ".mzml.gz"
        else:
            file_extension = filepath.suffix.lower()

        if file_extension == ".raw":
            # if BMSIOReader is None:
            #     raise ValueError(
            #         f"Unsupported file type: {file_extension}. Raw file reader is not installed."
            #     )
            # reader = BMSIOReader(filepath, server_addr=bmsio_server_addr)
            if acquisition_method is None:
                is_dda = "DIA"
            else:
                is_dda = acquisition_method
            reader = ThermoRawReader(filepath, dda=is_dda=="DDA")
        elif file_extension in [".mzml", ".mzml.gz"]:
            reader = MzmlFileReader(filepath)
        elif file_extension in [".hdf", ".h5"]:
            reader = HdfReader(filepath, run_name=run_name)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # _ = reader.open(filepath)
        return reader
