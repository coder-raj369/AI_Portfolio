"""Schedules are pure functions of the step index, so they get exact assertions."""

from __future__ import annotations

import numpy as np
import pytest

from nanoptim import ConstantLR, CosineWithWarmup, WarmupStableDecay


def test_constant_is_constant() -> None:
    s = ConstantLR(3e-4)
    assert {s(i) for i in range(100)} == {3e-4}


def test_cosine_warmup_reaches_peak_at_end_of_warmup() -> None:
    s = CosineWithWarmup(peak_lr=1.0, total_steps=1000, warmup_steps=100)
    assert s(0) == pytest.approx(0.01)  # 1/100 of peak, not zero
    assert s(99) == pytest.approx(1.0)
    assert s(100) == pytest.approx(1.0)


def test_cosine_decays_monotonically_to_min_lr() -> None:
    s = CosineWithWarmup(peak_lr=1.0, total_steps=1000, warmup_steps=100, min_lr=0.05)
    lrs = [s(i) for i in range(100, 1000)]
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:]))
    assert s(1000) == pytest.approx(0.05)
    assert s(5000) == pytest.approx(0.05)  # clamps past the end
    assert s(550) == pytest.approx(0.05 + 0.95 * 0.5, abs=0.02)  # ~halfway


def test_cosine_rejects_warmup_longer_than_run() -> None:
    with pytest.raises(ValueError):
        CosineWithWarmup(1.0, total_steps=10, warmup_steps=10)


def test_wsd_has_a_genuinely_flat_stable_phase() -> None:
    s = WarmupStableDecay(peak_lr=1.0, total_steps=1000, warmup_frac=0.02, decay_frac=0.1)
    assert s.warmup_steps == 20
    assert s.decay_start == 900
    stable = [s(i) for i in range(20, 900)]
    assert np.allclose(stable, 1.0)


def test_wsd_decays_to_min_lr_at_the_end() -> None:
    s = WarmupStableDecay(peak_lr=1.0, total_steps=1000, min_lr=0.0)
    assert s(900) == pytest.approx(1.0)
    assert s(999) < 0.11
    assert s(1000) == pytest.approx(0.0)
    decay = [s(i) for i in range(900, 1000)]
    assert all(b <= a + 1e-12 for a, b in zip(decay, decay[1:]))


def test_wsd_rejects_impossible_phase_fractions() -> None:
    with pytest.raises(ValueError):
        WarmupStableDecay(1.0, total_steps=100, warmup_frac=0.5, decay_frac=0.6)


def test_schedules_never_return_negative_or_nan() -> None:
    for s in (
        CosineWithWarmup(1e-3, 500, 50, min_lr=1e-5),
        WarmupStableDecay(1e-3, 500),
        ConstantLR(1e-3),
    ):
        values = [s(i) for i in range(600)]
        assert all(v >= 0 and np.isfinite(v) for v in values)
