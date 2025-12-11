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

"""Minimal LR schedulers: cosine with warmup & cosine with hard restarts (warmup)."""

import math
from functools import partial
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

__all__ = [
    "get_cosine_schedule_with_warmup",
    "get_cosine_with_hard_restarts_schedule_with_warmup",
]

# ---- Internal lambda helpers -------------------------------------------------


def _cosine_warmup_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
) -> float:
    """Cosine annealing with linear warmup (no restarts)."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    # half-cosine from 1.0 → 0.0 when num_cycles=0.5
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )


def _cosine_hard_restarts_warmup_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
) -> float:
    """Cosine annealing with linear warmup + hard restarts."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    if progress >= 1.0:
        return 0.0
    # progress in [0,1); cycles wrap by modulo
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
    )


# ---- Public factory functions ------------------------------------------------


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Cosine annealing from initial LR → 0 with linear warmup.

    Args:
        optimizer: torch optimizer to schedule.
        num_warmup_steps: warmup steps (linear 0 → initial LR).
        num_training_steps: total training steps.
        num_cycles: number of cosine cycles (default 0.5 = single half-cycle).
        last_epoch: PyTorch scheduler arg for resuming.

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    lr_lambda = partial(
        _cosine_warmup_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Cosine annealing with linear warmup and hard restarts.

    Args:
        optimizer: torch optimizer to schedule.
        num_warmup_steps: warmup steps (linear 0 → initial LR).
        num_training_steps: total training steps.
        num_cycles: number of hard restarts (integer).
        last_epoch: PyTorch scheduler arg for resuming.

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    lr_lambda = partial(
        _cosine_hard_restarts_warmup_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
