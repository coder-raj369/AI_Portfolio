"""Initialisation: measured variance against the closed-form target."""

from __future__ import annotations

import numpy as np
import pytest

from nnprim.init import (
    kaiming_normal,
    kaiming_uniform,
    residual_scaled_normal,
    truncated_normal,
    xavier_normal,
    xavier_uniform,
)

RNG = np.random.default_rng(21)
SHAPE = (512, 256)


def test_xavier_uniform_variance_matches_2_over_fan_sum() -> None:
    w = xavier_uniform(SHAPE, RNG)
    expected = 2.0 / (SHAPE[0] + SHAPE[1])
    assert w.var() == pytest.approx(expected, rel=0.05)


def test_xavier_normal_variance_matches_2_over_fan_sum() -> None:
    w = xavier_normal(SHAPE, RNG)
    assert w.var() == pytest.approx(2.0 / (SHAPE[0] + SHAPE[1]), rel=0.05)


def test_kaiming_variance_is_2_over_fan_in() -> None:
    for fn in (kaiming_uniform, kaiming_normal):
        w = fn(SHAPE, RNG)
        assert w.var() == pytest.approx(2.0 / SHAPE[0], rel=0.05), fn.__name__


def test_kaiming_preserves_activation_variance_through_relu() -> None:
    """The actual justification for the factor of 2, measured over 8 layers."""
    x = RNG.standard_normal((2048, 256))
    h = x
    for _ in range(8):
        w = kaiming_normal((256, 256), RNG)
        h = np.maximum(h @ w, 0.0)
    # ReLU halves the variance; Kaiming's 2/fan_in restores it, so the variance
    # of the *pre-activation* stays O(1) instead of decaying like 2^-L.
    assert 0.2 < h.var() < 5.0


def test_xavier_would_decay_variance_through_relu() -> None:
    x = RNG.standard_normal((2048, 256))
    h = x
    for _ in range(8):
        w = xavier_normal((256, 256), RNG)
        h = np.maximum(h @ w, 0.0)
    assert h.var() < 0.2  # ~2^-8 of the Kaiming path


def test_truncated_normal_respects_its_bound_and_shrinks_the_std() -> None:
    """Truncation is not free: clipping at +-2 sigma leaves a realised std of
    0.8796 sigma, not sigma. This is why a config's `initializer_range` is not
    the standard deviation of the weights that actually get created."""
    w = truncated_normal((1000, 100), RNG, std=0.02, clip=2.0)
    assert np.abs(w).max() <= 2.0 * 0.02 + 1e-12
    assert w.std() == pytest.approx(0.02 * 0.8796, rel=0.02)


def test_residual_scaling_shrinks_std_by_sqrt_2n() -> None:
    n_layers = 12
    w = residual_scaled_normal((512, 512), RNG, n_layers=n_layers, std=0.02)
    assert w.std() == pytest.approx(0.02 / np.sqrt(2 * n_layers), rel=0.05)


def test_residual_scaling_keeps_stream_variance_bounded_with_depth() -> None:
    """Without the 1/sqrt(2N) factor, residual-stream variance grows ~linearly
    in depth; with it, the stream stays O(1) at the final layer."""
    d, n_layers = 256, 24
    stream_scaled = RNG.standard_normal((256, d))
    stream_plain = stream_scaled.copy()
    for _ in range(n_layers):
        w_s = residual_scaled_normal((d, d), RNG, n_layers=n_layers, std=0.02)
        w_p = RNG.normal(0.0, 0.02, size=(d, d))
        stream_scaled = stream_scaled + stream_scaled @ w_s
        stream_plain = stream_plain + stream_plain @ w_p
    assert stream_scaled.var() < stream_plain.var()


def test_init_requires_at_least_two_dimensions() -> None:
    with pytest.raises(ValueError):
        xavier_uniform((10,), RNG)
