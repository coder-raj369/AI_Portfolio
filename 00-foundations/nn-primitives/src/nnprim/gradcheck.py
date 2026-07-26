"""Central-difference gradient checking for the hand-derived backward passes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numeric_grad(scalar_fn: Callable[[], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of `scalar_fn()` w.r.t. every element of `x`.

    `scalar_fn` must re-read `x` on each call (close over it; do not copy it),
    since this perturbs `x` in place.
    """
    grad = np.zeros_like(x)
    flat = x.reshape(-1)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = scalar_fn()
        flat[i] = original - eps
        minus = scalar_fn()
        flat[i] = original
        grad.reshape(-1)[i] = (plus - minus) / (2 * eps)
    return grad


def max_abs_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """Largest absolute discrepancy - the number quoted in the README tables."""
    return float(np.abs(analytic - numeric).max())
