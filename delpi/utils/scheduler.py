# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors
# and The HuggingFace Inc. team.
# Modifications Copyright 2025 Jungkap Park
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal LR schedulers: cosine with warmup & cosine with hard restarts (warmup).

DelPi-local implementation (no HF dependency), extended to support:
- min_lr (absolute) and/or min_lr_rate (ratio)
- param-group-safe behavior (works even if groups have different base lr)
"""

from __future__ import annotations

import math
from functools import partial
from typing import List, Optional, Sequence, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

__all__ = [
    "get_cosine_schedule_with_warmup",
    "get_cosine_with_hard_restarts_schedule_with_warmup",
]

MinLRType = Union[float, Sequence[float], None]


# ---- Internal helpers --------------------------------------------------------


def _as_min_lr_rates(
    optimizer: Optimizer,
    *,
    min_lr: MinLRType = None,
    min_lr_rate: Optional[float] = None,
) -> List[float]:
    """Return per-param-group min_lr_rate in [0, 1].

    Rules:
      - If both provided -> error.
      - If min_lr_rate provided -> same ratio applied to all groups.
      - Else if min_lr provided:
          * float -> same absolute min_lr for all groups
          * sequence -> per-group absolute min_lr
        converted to ratio per group: min_lr_i / base_lr_i.
      - Else -> default 0.0 (classic cosine to zero).
    """
    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set.")

    n_groups = len(optimizer.param_groups)

    # ratio provided
    if min_lr_rate is not None:
        r = float(min_lr_rate)
        r = min(max(r, 0.0), 1.0)
        return [r] * n_groups

    # absolute provided
    if min_lr is None:
        return [0.0] * n_groups

    if isinstance(min_lr, (int, float)):
        min_lrs = [float(min_lr)] * n_groups
    else:
        min_lrs = [float(x) for x in min_lr]
        if len(min_lrs) != n_groups:
            raise ValueError(
                f"min_lr has {len(min_lrs)} values but optimizer has {n_groups} param_groups."
            )

    rates: List[float] = []
    for g, mlr in zip(optimizer.param_groups, min_lrs):
        base_lr = float(g.get("lr", 0.0))
        if base_lr <= 0.0:
            rates.append(0.0)
            continue
        r = mlr / base_lr
        r = min(max(float(r), 0.0), 1.0)
        rates.append(r)

    return rates


# ---- Internal lambda helpers -------------------------------------------------


def _cosine_warmup_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_rate: float,
) -> float:
    """Cosine annealing with linear warmup (no restarts), scaled to [min_lr_rate, 1]."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )

    if progress >= 1.0:
        return float(min_lr_rate)

    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1.0 - float(min_lr_rate)) + float(min_lr_rate)
    return max(0.0, float(factor))


def _cosine_hard_restarts_warmup_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
    min_lr_rate: float,
) -> float:
    """Cosine annealing with linear warmup + hard restarts, scaled to [min_lr_rate, 1]."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )

    if progress >= 1.0:
        return float(min_lr_rate)

    # progress in [0,1); cycles wrap by modulo
    factor = 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
    factor = factor * (1.0 - float(min_lr_rate)) + float(min_lr_rate)
    return max(0.0, float(factor))


# ---- Public factory functions ------------------------------------------------


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    *,
    min_lr: MinLRType = None,
    min_lr_rate: float | None = None,
) -> LambdaLR:
    """Cosine annealing from initial LR → min_lr (default 0) with linear warmup.

    Args:
        optimizer: torch optimizer to schedule.
        num_warmup_steps: warmup steps (linear 0 → initial LR).
        num_training_steps: total training steps.
        num_cycles: number of cosine cycles (default 0.5 = single half-cycle).
        last_epoch: PyTorch scheduler arg for resuming.
        min_lr: absolute minimum LR (float) or per-param-group list; default None (=0).
        min_lr_rate: ratio min LR to initial LR. If set, min_lr must be None.

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    min_lr_rates = _as_min_lr_rates(optimizer, min_lr=min_lr, min_lr_rate=min_lr_rate)

    lr_lambdas = [
        partial(
            _cosine_warmup_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            min_lr_rate=r,
        )
        for r in min_lr_rates
    ]
    return LambdaLR(optimizer, lr_lambdas, last_epoch=last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
    *,
    min_lr: MinLRType = None,
    min_lr_rate: float | None = None,
) -> LambdaLR:
    """Cosine annealing with linear warmup and hard restarts, from initial LR → min_lr (default 0).

    Args:
        optimizer: torch optimizer to schedule.
        num_warmup_steps: warmup steps (linear 0 → initial LR).
        num_training_steps: total training steps.
        num_cycles: number of hard restarts (integer, >=1).
        last_epoch: PyTorch scheduler arg for resuming.
        min_lr: absolute minimum LR (float) or per-param-group list; default None (=0).
        min_lr_rate: ratio min LR to initial LR. If set, min_lr must be None.

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    if num_cycles < 1:
        raise ValueError("num_cycles must be >= 1 for hard restarts schedule.")

    min_lr_rates = _as_min_lr_rates(optimizer, min_lr=min_lr, min_lr_rate=min_lr_rate)

    lr_lambdas = [
        partial(
            _cosine_hard_restarts_warmup_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            min_lr_rate=r,
        )
        for r in min_lr_rates
    ]
    return LambdaLR(optimizer, lr_lambdas, last_epoch=last_epoch)
