"""Initialisation schemes, plus the residual-stack rescaling that GPT-2 introduced.

Initialisation is where "it trains but worse" bugs live: the wrong fan
convention costs a factor of ~2 in activation variance per layer, which
compounds to 2^L across a deep stack.
"""

from __future__ import annotations

import numpy as np


def _fans(shape: tuple[int, ...]) -> tuple[int, int]:
    """Return `(fan_in, fan_out)` for a weight of shape `(in, out)` or conv-like."""
    if len(shape) < 2:
        raise ValueError("initialisation needs at least a 2-d weight")
    receptive = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    return shape[0] * receptive, shape[1] * receptive


def xavier_uniform(shape: tuple[int, ...], rng: np.random.Generator, gain: float = 1.0):
    """Glorot: bound = gain * sqrt(6 / (fan_in + fan_out)). For tanh/sigmoid nets."""
    fan_in, fan_out = _fans(shape)
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=shape)


def xavier_normal(shape: tuple[int, ...], rng: np.random.Generator, gain: float = 1.0):
    fan_in, fan_out = _fans(shape)
    return rng.normal(0.0, gain * np.sqrt(2.0 / (fan_in + fan_out)), size=shape)


def kaiming_uniform(shape: tuple[int, ...], rng: np.random.Generator, a: float = 0.0):
    """He, for ReLU-family activations: bound = sqrt(6 / ((1 + a^2) * fan_in)).

    `a` is the negative slope of the LeakyReLU that follows; `a=0` is plain ReLU.
    The extra factor of 2 relative to Xavier compensates for ReLU zeroing half
    the units, which halves the variance of the layer output.
    """
    fan_in, _ = _fans(shape)
    bound = np.sqrt(6.0 / ((1 + a * a) * fan_in))
    return rng.uniform(-bound, bound, size=shape)


def kaiming_normal(shape: tuple[int, ...], rng: np.random.Generator, a: float = 0.0):
    fan_in, _ = _fans(shape)
    return rng.normal(0.0, np.sqrt(2.0 / ((1 + a * a) * fan_in)), size=shape)


def truncated_normal(
    shape: tuple[int, ...], rng: np.random.Generator, std: float = 0.02, clip: float = 2.0
):
    """Normal truncated at +-`clip` std, resampled rather than clamped.

    Clamping piles mass on the boundary; resampling keeps the distribution
    smooth. `std=0.02` is the GPT-2/GPT-NeoX convention.
    """
    out = rng.normal(0.0, std, size=shape)
    for _ in range(100):
        bad = np.abs(out) > clip * std
        if not bad.any():
            break
        out[bad] = rng.normal(0.0, std, size=int(bad.sum()))
    return out


def residual_scaled_normal(
    shape: tuple[int, ...], rng: np.random.Generator, n_layers: int, std: float = 0.02
):
    r"""GPT-2's residual rescaling: divide the output-projection std by
    :math:`\sqrt{2 N}` for an `N`-layer stack.

    Each of the `2N` residual branches (attention + MLP per layer) adds variance
    to the stream. Without this, activation variance grows linearly with depth
    and the final-layer logits blow up before training even starts.
    """
    return rng.normal(0.0, std / np.sqrt(2.0 * n_layers), size=shape)
