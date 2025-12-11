import os
from pathlib import Path
from typing import Type, Optional, Dict, Any

import torch
from torch.serialization import MAP_LOCATION
from pytorch_lightning import LightningModule


def load_latest_checkpoint(
    model_class: Type[LightningModule],
    version_dir: Path,
    map_location: MAP_LOCATION = None,
    **kwargs,
) -> Optional[LightningModule]:
    """
    Load the most recently modified checkpoint from a specific version directory under lightning_logs.

    Args:
        version_dir (Path): model directory where a lightning model is stored
        map_location (torch.device): map location
    Returns:
        Loaded LightningModule or None if no checkpoint is found.
    """

    if not version_dir.exists():
        raise FileNotFoundError(f"Directory {version_dir} does not exist.")

    # Get all .ckpt files in the checkpoint directory
    ckpt_files = list((f for f in version_dir.glob("**/*.ckpt") if f.stem != "last"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {version_dir}")

    # Sort by modification time (latest last)
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)

    return model_class.load_from_checkpoint(
        latest_ckpt, map_location=map_location, **kwargs
    )


def load_latest_checkpoint_from_version(
    model_class: Type[LightningModule],
    version: int,
    logs_root: str = "lightning_logs",
    map_location: MAP_LOCATION = None,
    **kwargs,
) -> Optional[LightningModule]:

    version_dir = Path(logs_root) / f"version_{version}"

    return load_latest_checkpoint(model_class, version_dir, map_location, **kwargs)


def load_best_checkpoint(
    version: int,
    logs_root: str = "lightning_logs",
    map_location: MAP_LOCATION = None,
    **kwargs,
) -> Dict[Any, Any]:

    version_dir = Path(logs_root) / f"version_{version}"

    if not version_dir.exists():
        raise FileNotFoundError(f"Directory {version_dir} does not exist.")

    # Get all .ckpt files in the checkpoint directory
    ckpt_files = list((f for f in version_dir.glob("**/*.ckpt") if f.stem != "last"))
    if len(ckpt_files) != 1:
        raise FileNotFoundError(
            f"Cannot find a single checkpoint file in {version_dir}"
        )

    return torch.load(
        ckpt_files[0], weights_only=True, map_location=map_location, **kwargs
    )
