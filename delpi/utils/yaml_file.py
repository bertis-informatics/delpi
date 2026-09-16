import yaml
import h5py
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union


def load_yaml(filename) -> dict:
    with open(filename) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    return settings


def save_yaml(filename, settings):
    with open(filename, "w") as file:
        yaml.dump(settings, file, sort_keys=False)


def save_yaml_to_hdf(hdf_file_path: str, key: str, yaml_dict: dict):
    with h5py.File(hdf_file_path, "a") as f:
        if key in f:
            del f[key]
        yaml_str = yaml.dump(yaml_dict, sort_keys=False)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(key, data=yaml_str, dtype=dt)


def load_yaml_to_hdf(hdf_file_path: str, key: str) -> dict:
    with h5py.File(hdf_file_path, "r") as f:
        yaml_str = f[key][()].decode() if isinstance(f[key][()], bytes) else f[key][()]
    return yaml.safe_load(yaml_str)


def resolve_input_files(
    entries: Union[str, List[str]],
    allowed_extensions: Optional[Tuple[str, ...]] = None,
) -> List[Path]:
    """Expand path/glob entries (explicit paths and/or glob patterns, including `**`) into a sorted, deduplicated list of absolute file paths."""
    if isinstance(entries, str):
        entries = [entries]

    resolved = set()
    for entry in entries:
        matches = glob(entry, recursive=True)
        if not matches:
            raise ValueError(f"No files matched input entry: {entry!r}")

        for match in matches:
            path = Path(match).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"Resolved input path is not a file: {path}")
            if allowed_extensions and not str(path).lower().endswith(allowed_extensions):
                raise ValueError(f"Unsupported file type for {path}")
            resolved.add(path)

    if not resolved:
        raise ValueError("No input files resolved from configuration.")

    return sorted(resolved)
