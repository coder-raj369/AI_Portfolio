"""Learning-rate schedules, including the warmup-stable-decay shape used by
most 2024-2026 open LLM pretraining runs.

Schedules are plain callables of the step index, so they can be composed with
any optimiser via `opt.set_lr(schedule(step))` and unit-tested without training
anything.
"""

from __future__ import annotations

import math


class Schedule:
    """Base class: a callable mapping a 0-based step index to a learning rate."""

    def __call__(self, step: int) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class ConstantLR(Schedule):
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def __call__(self, step: int) -> float:
        return self.lr


class CosineWithWarmup(Schedule):
    """Linear warmup, then cosine decay from `peak_lr` to `min_lr`.

    Warmup exists because the adaptive denominator in Adam is badly estimated
    for the first few hundred steps; starting at the peak LR is the classic way
    to blow up a transformer run in the first 100 steps.
    """

    def __init__(
        self, peak_lr: float, total_steps: int, warmup_steps: int = 0, min_lr: float = 0.0
    ) -> None:
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            # step+1 so that step 0 is not a dead step with lr exactly 0
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine


class WarmupStableDecay(Schedule):
    """Warmup -> constant "stable" phase -> short decay (WSD / trapezoidal).

    Preferred over cosine for open pretraining runs because the stable phase
    yields usable intermediate checkpoints: you can branch a decay from any
    point instead of committing to a total token budget up front.
    """

    def __init__(
        self,
        peak_lr: float,
        total_steps: int,
        warmup_frac: float = 0.02,
        decay_frac: float = 0.1,
        min_lr: float = 0.0,
    ) -> None:
        if warmup_frac + decay_frac >= 1.0:
            raise ValueError("warmup_frac + decay_frac must be < 1")
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = max(int(total_steps * warmup_frac), 1)
        self.decay_steps = max(int(total_steps * decay_frac), 1)
        self.min_lr = min_lr

    @property
    def decay_start(self) -> int:
        return self.total_steps - self.decay_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step < self.decay_start:
            return self.peak_lr
        if step >= self.total_steps:
            return self.min_lr
        # 1 - sqrt decay: sharper than linear, which empirically recovers more
        # of the loss drop in a short decay window.
        progress = (step - self.decay_start) / self.decay_steps
        return self.min_lr + (self.peak_lr - self.min_lr) * (1.0 - math.sqrt(progress))
