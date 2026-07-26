"""Normalisation: forward properties and hand-derived backward passes."""

from __future__ import annotations

import numpy as np
import pytest

from nnprim.gradcheck import numeric_grad
from nnprim.norm import layer_norm, layer_norm_backward, rms_norm, rms_norm_backward

RNG = np.random.default_rng(4)


def test_layer_norm_output_is_zero_mean_unit_variance() -> None:
    x = RNG.standard_normal((5, 32)) * 7.0 + 3.0
    y, _ = layer_norm(x)
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(y.std(axis=-1), 1.0, atol=1e-4)


def test_rms_norm_does_not_centre() -> None:
    """The defining difference from LayerNorm: a constant offset survives."""
    x = np.full((1, 8), 5.0)
    y, _ = rms_norm(x)
    assert y.mean() == pytest.approx(1.0, rel=1e-4)  # not 0


def test_rms_norm_is_scale_invariant() -> None:
    x = RNG.standard_normal((4, 16))
    a, _ = rms_norm(x)
    b, _ = rms_norm(x * 137.0)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_layer_norm_backward_matches_finite_differences() -> None:
    x = RNG.standard_normal((4, 12))
    g = RNG.standard_normal((4, 12))
    gamma, beta = RNG.standard_normal(12), RNG.standard_normal(12)
    _, cache = layer_norm(x, gamma, beta)
    dx, dgamma, dbeta = layer_norm_backward(g, cache)

    def loss() -> float:
        return float((layer_norm(x, gamma, beta)[0] * g).sum())

    assert np.abs(dx - numeric_grad(loss, x)).max() < 1e-7
    assert np.abs(dgamma - numeric_grad(loss, gamma)).max() < 1e-7
    assert np.abs(dbeta - numeric_grad(loss, beta)).max() < 1e-7


def test_dropping_the_correction_terms_gives_a_wrong_gradient() -> None:
    """Guards against the classic simplification `dx = g / sigma`."""
    x = RNG.standard_normal((3, 10))
    g = RNG.standard_normal((3, 10))
    _, cache = layer_norm(x)
    dx, _, _ = layer_norm_backward(g, cache)
    naive = g * cache["inv"]
    assert np.abs(dx - naive).max() > 0.05


def test_layer_norm_gradient_is_orthogonal_to_the_removed_directions() -> None:
    """LayerNorm output cannot change under a shift or rescale of the input, so
    dL/dx must have zero component along both 1 and xhat."""
    x = RNG.standard_normal((6, 20))
    g = RNG.standard_normal((6, 20))
    # Orthogonality along xhat is exact only as eps -> 0: a finite eps makes
    # mean(xhat^2) = var/(var+eps) < 1, leaving a residual of order eps.
    _, cache = layer_norm(x, eps=1e-12)
    dx, _, _ = layer_norm_backward(g, cache)
    np.testing.assert_allclose(dx.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose((dx * cache["xhat"]).sum(axis=-1), 0.0, atol=1e-9)


def test_rms_norm_backward_matches_finite_differences() -> None:
    x = RNG.standard_normal((4, 12))
    g = RNG.standard_normal((4, 12))
    gamma = RNG.standard_normal(12)
    _, cache = rms_norm(x, gamma)
    dx, dgamma = rms_norm_backward(g, cache)

    def loss() -> float:
        return float((rms_norm(x, gamma)[0] * g).sum())

    assert np.abs(dx - numeric_grad(loss, x)).max() < 1e-7
    assert np.abs(dgamma - numeric_grad(loss, gamma)).max() < 1e-7


def test_both_norms_survive_an_all_zero_row() -> None:
    x = np.zeros((2, 8))
    assert np.all(np.isfinite(layer_norm(x)[0]))
    assert np.all(np.isfinite(rms_norm(x)[0]))


def test_norms_work_on_3d_activations() -> None:
    x = RNG.standard_normal((2, 7, 16))
    gamma = RNG.standard_normal(16)
    y, cache = rms_norm(x, gamma)
    assert y.shape == x.shape
    dx, dgamma = rms_norm_backward(np.ones_like(y), cache)
    assert dx.shape == x.shape and dgamma.shape == (16,)
