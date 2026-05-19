"""
Device-aware inference/AMP/SDPA context helpers.

- CPU: autocast(bfloat16) by default, SDPA -> MATH
- CUDA: autocast(float16) by default, SDPA -> EFFICIENT_ATTENTION
- MPS: autocast(float16) by default, SDPA -> MATH (safe; optimized CUDA backends not available)

This module returns context managers you can use in `with ...:` blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext, ExitStack, AbstractContextManager
from typing import Iterable, Optional, Union

import torch


# ---- Optional SDPA imports (PyTorch 2.x) ----
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend  # type: ignore

    _HAS_SDPA = True
except Exception:  # pragma: no cover
    sdpa_kernel = None  # type: ignore
    SDPBackend = None  # type: ignore
    _HAS_SDPA = False


DeviceLike = Union[torch.device, str]


@dataclass(frozen=True)
class InferenceContexts:
    """
    Individual context managers plus a convenient `.all()` combined context.
    Use either:
        with ctx.all(): ...
    or
        with ctx.inference, ctx.amp, ctx.sdpa: ...
    """

    inference: AbstractContextManager
    amp: AbstractContextManager
    sdpa: AbstractContextManager

    def all(self) -> AbstractContextManager:
        """
        Combine inference + amp + sdpa into a single context manager.
        """
        stack = ExitStack()
        stack.enter_context(self.inference)
        stack.enter_context(self.amp)
        stack.enter_context(self.sdpa)
        return stack


def _to_device(device: DeviceLike) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _default_amp_dtype(device_type: str) -> Optional[torch.dtype]:
    # Conservative, widely-used defaults
    if device_type == "cuda":
        return torch.float16
    if device_type == "mps":
        return torch.float16
    if device_type == "cpu":
        return torch.bfloat16
    return None


def _pick_sdpa_backends(device_type: str) -> Optional[Iterable]:
    """
    Return a backend preference list for sdpa_kernel([...]) or None (no SDPA ctx).
    NOTE: Optimized backends are CUDA-centric; for MPS/CPU we keep MATH as safest.
    """
    if not _HAS_SDPA:
        return None

    # CUDA: prefer optimized SDPA backends. Both FLASH and EFFICIENT produce
    # results very close to fp32 training (~0.2% relative error), while the
    # legacy manual-attention path (timm <1.0) diverges ~5% from fp32.
    if device_type == "cuda":
        preferred = []
        if hasattr(SDPBackend, "FLASH_ATTENTION"):
            preferred.append(SDPBackend.FLASH_ATTENTION)
        if hasattr(SDPBackend, "EFFICIENT_ATTENTION"):
            preferred.append(SDPBackend.EFFICIENT_ATTENTION)
        if hasattr(SDPBackend, "MATH"):
            preferred.append(SDPBackend.MATH)
        return preferred if preferred else None

    # CPU/MPS/others: stick to MATH
    if hasattr(SDPBackend, "MATH"):
        return [SDPBackend.MATH]
    return None


def _pick_explicit_sdpa_backend(
    device_type: str,
    sdpa_backend: str,
) -> Optional[Iterable]:
    """
    Return a single explicitly requested backend for sdpa_kernel([...]).

    Supported names (case-insensitive):
        - "flash"
        - "efficient"
        - "math"
    """
    if not _HAS_SDPA:
        return None

    name = sdpa_backend.strip().lower()
    backend_name_map = {
        "flash": "FLASH_ATTENTION",
        "efficient": "EFFICIENT_ATTENTION",
        "math": "MATH",
    }

    if name not in backend_name_map:
        raise ValueError(
            f"Invalid sdpa_backend={sdpa_backend!r}. "
            "Use one of: flash, efficient, math."
        )

    enum_name = backend_name_map[name]
    if not hasattr(SDPBackend, enum_name):
        raise ValueError(
            f"Requested SDPA backend '{name}' is not available on this torch build."
        )

    backend = getattr(SDPBackend, enum_name)

    # Keep CPU/MPS safe by disallowing CUDA-only kernels there.
    if device_type != "cuda" and enum_name != "MATH":
        raise ValueError(
            f"SDPA backend '{name}' requires CUDA device, got device_type={device_type!r}."
        )

    return [backend]


def make_inference_contexts(
    device: DeviceLike,
    *,
    enable_inference_mode: bool = True,
    enable_autocast: bool = True,
    autocast_dtype: Optional[torch.dtype] = None,
    enable_sdpa_kernel: bool = True,
    sdpa_backend: Optional[str] = None,
) -> InferenceContexts:
    """
    Build device-aware context managers.

    Parameters
    ----------
    device:
        torch.device or string ("cpu", "cuda", "cuda:0", "mps", ...)
    enable_inference_mode:
        Use torch.inference_mode() if True, else nullcontext().
    enable_autocast:
        Enable AMP autocast if True.
    autocast_dtype:
        If None, picks a safe default per device (cuda/mps: fp16, cpu: bf16).
        If you pass fp16 on CPU, some ops may fail depending on your model/op set.
    enable_sdpa_kernel:
        If True and sdpa_kernel is available, sets a safe backend preference.
    sdpa_backend:
        Optional explicit SDPA backend name: "flash", "efficient", or "math".
        If None, uses the existing preferred-order selection by device.

    Returns
    -------
    InferenceContexts
    """
    dev = _to_device(device)
    dt = dev.type  # "cpu" | "cuda" | "mps" | ...

    # 1) inference mode
    inference_ctx = torch.inference_mode() if enable_inference_mode else nullcontext()

    # 2) autocast
    if enable_autocast:
        dtype = autocast_dtype if autocast_dtype is not None else _default_amp_dtype(dt)
        if dtype is None:
            amp_ctx = nullcontext()
        else:
            # torch.autocast supports device_type="cuda"/"cpu"/"mps" in modern PyTorch.
            amp_ctx = torch.autocast(device_type=dt, dtype=dtype)
    else:
        amp_ctx = nullcontext()

    # 3) sdpa kernel preference
    if enable_sdpa_kernel and _HAS_SDPA:
        if sdpa_backend is None:
            backends = _pick_sdpa_backends(dt)
        else:
            backends = _pick_explicit_sdpa_backend(dt, sdpa_backend)
        sdpa_ctx = sdpa_kernel(backends) if backends else nullcontext()
    else:
        sdpa_ctx = nullcontext()

    return InferenceContexts(inference=inference_ctx, amp=amp_ctx, sdpa=sdpa_ctx)
