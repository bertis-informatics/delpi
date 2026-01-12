# device_contexts.py
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

    # CUDA: prefer efficient attention (you may add FLASH_ATTENTION if you want)
    if device_type == "cuda":
        # Some installs may not have EFFICIENT_ATTENTION; guard dynamically.
        preferred = []
        if hasattr(SDPBackend, "EFFICIENT_ATTENTION"):
            preferred.append(SDPBackend.EFFICIENT_ATTENTION)
        # Fallback to MATH if needed
        if hasattr(SDPBackend, "MATH"):
            preferred.append(SDPBackend.MATH)
        return preferred if preferred else None

    # CPU/MPS/others: stick to MATH
    if hasattr(SDPBackend, "MATH"):
        return [SDPBackend.MATH]
    return None


def make_inference_contexts(
    device: DeviceLike,
    *,
    enable_inference_mode: bool = True,
    enable_autocast: bool = True,
    autocast_dtype: Optional[torch.dtype] = None,
    enable_sdpa_kernel: bool = True,
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
        backends = _pick_sdpa_backends(dt)
        sdpa_ctx = sdpa_kernel(backends) if backends else nullcontext()
    else:
        sdpa_ctx = nullcontext()

    return InferenceContexts(inference=inference_ctx, amp=amp_ctx, sdpa=sdpa_ctx)
