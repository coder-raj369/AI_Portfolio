"""Finite-difference gradient checking - the reason to trust this engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from nanograd.tensor import Tensor


def numeric_grad(fn: Callable[[], Tensor], tensor: Tensor, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of `fn()` w.r.t. every element of `tensor`.

    `fn` must be a closure that re-runs the forward pass and returns a scalar
    `Tensor`, reading `tensor.data` each time.
    """
    grad = np.zeros_like(tensor.data)
    flat = tensor.data.reshape(-1)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = fn().item()
        flat[i] = original - eps
        minus = fn().item()
        flat[i] = original
        grad.reshape(-1)[i] = (plus - minus) / (2 * eps)
    return grad


def gradcheck(
    fn: Callable[[], Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
    tol: float = 1e-5,
) -> float:
    """Compare analytic gradients against central differences.

    Returns:
        The largest relative error across all inputs.

    Raises:
        AssertionError: if that error exceeds `tol`.
    """
    out = fn()
    for t in inputs:
        t.zero_grad()
    out.backward()
    worst = 0.0
    for t in inputs:
        analytic = np.zeros_like(t.data) if t.grad is None else t.grad
        numeric = numeric_grad(fn, t, eps=eps)
        denom = np.maximum(1.0, np.maximum(np.abs(analytic), np.abs(numeric)))
        worst = max(worst, float(np.max(np.abs(analytic - numeric) / denom)))
    if worst > tol:
        raise AssertionError(f"gradcheck failed: max relative error {worst:.3e} > tol {tol:.1e}")
    return worst
