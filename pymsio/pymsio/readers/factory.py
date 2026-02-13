from pathlib import Path
from typing import Union

from .base import MassSpecFileReader
from .thermo import ThermoRawReader
from .mzml import MzmlFileReader


class ReaderFactory:
    supported_file_extensions = [".raw", ".mzml", ".mzml.gz"]

    @classmethod
    def _normalize_ext(cls, path: Path) -> str:
        name = path.name.lower()
        if name.endswith(".mzml.gz"):
            return ".mzml.gz"
        return path.suffix.lower()

    @classmethod
    def get_reader(
        cls,
        filepath: Union[str, Path],
        show_progress: bool = False,
    ) -> MassSpecFileReader:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        ext = cls._normalize_ext(filepath)

        if ext not in cls.supported_file_extensions:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(cls.supported_file_extensions)}"
            )

        if ext == ".raw":
            reader = ThermoRawReader(filepath, show_progress=show_progress)
        elif ext in (".mzml", ".mzml.gz"):
            reader = MzmlFileReader(filepath, show_progress=show_progress)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return reader
