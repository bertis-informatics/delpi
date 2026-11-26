"""Export and load inference-only checkpoints without Lightning.

The file is a plain dict::

    {
        "model_state_dict": ...,   # weights for load_state_dict
        "hyper_parameters": ...,   # kwargs for the target class __init__
        "meta": ...,               # free-form metadata (versions, timestamps, ...)
    }

Used by DelPiModel, DelPiClassifier, Ms2SpectrumPredictor and
RetentionTimePredictor.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Type, TypeVar
import warnings

import torch
import torch.nn as nn

T = TypeVar("T", bound=nn.Module)


def save_model(
    model: nn.Module,
    save_path: str | Path,
    *,
    inference_cls: Type[nn.Module] | None = None,
    model_version: str = "v1.0",
    extra_meta: dict | None = None,
) -> None:
    """Export ``model`` weights, hyperparameters and meta to ``save_path``.

    Args:
        model: a LightningModule, or any ``nn.Module`` with ``.hparams``.
        inference_cls: inference-only class whose ``__init__`` kwargs and
            ``state_dict`` keys are kept. None uses ``type(model)`` and saves
            the full ``state_dict``.
        extra_meta: merged into ``meta`` (e.g. acquisition_mode, token schemas).
    """
    from delpi import __version__ as code_version

    target_cls = inference_cls or type(model)
    init_argnames = target_cls.__init__.__code__.co_varnames
    hparams = {k: v for k, v in model.hparams.items() if k in init_argnames}

    if inference_cls is not None:
        ref_keys = set(inference_cls(**hparams).state_dict().keys())
        state_dict = {k: v for k, v in model.state_dict().items() if k in ref_keys}
    else:
        state_dict = dict(model.state_dict())

    meta = {
        "model_name": target_cls.__name__,
        "model_version": model_version,
        "code_version": code_version,
        "torch_version": str(torch.__version__),
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)

    payload = {
        "model_state_dict": state_dict,
        "hyper_parameters": hparams,
        "meta": meta,
    }
    torch.save(payload, save_path)


def load_model(cls: Type[T], path: str | Path) -> T:
    """Load a model exported by :func:`save_model`, in eval mode on CPU, with
    ``meta`` attached as an attribute.

    Raises:
        ValueError: ``meta["model_name"]`` differs from ``cls.__name__``.
    """
    payload = torch.load(Path(path), weights_only=True)
    meta = payload.get("meta", {})

    saved_name = meta.get("model_name")
    if saved_name is not None and saved_name != cls.__name__:
        raise ValueError(
            f"Model class mismatch loading {path!s}: file was exported as "
            f"{saved_name!r} but is being loaded as {cls.__name__!r}."
        )
    if saved_name is None:
        warnings.warn(
            f"Loading {path!s} as {cls.__name__}: 'model_name' missing from "
            "meta; cannot verify class match.",
            stacklevel=2,
        )

    model = cls(**payload["hyper_parameters"])
    model.load_state_dict(payload["model_state_dict"])
    model.meta = meta
    model.eval()
    return model
