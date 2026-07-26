"""Stable activations and losses, with the naive versions kept for comparison.

The `naive_*` functions are not dead code: `experiments/stability_report.py`
imports them to measure the exact logit magnitude at which each dtype breaks,
which is the whole point of the module.
"""

from __future__ import annotations

import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """`log(sum(exp(x)))` computed as `m + log(sum(exp(x - m)))`.

    Subtracting the max means the largest argument to `exp` is 0, so the result
    is exact wherever it is representable at all.
    """
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)  # all -inf row (fully masked) must not become nan
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def naive_logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """The textbook formula. Overflows at x ~ 89 (float32) / 710 (float64)."""
    return np.log(np.sum(np.exp(x), axis=axis))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax: shift by the max, then normalise."""
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def naive_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def naive_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """`log(softmax(x))`: loses precision from below as well as above -
    once a probability underflows to 0, the log is -inf."""
    return np.log(naive_softmax(x, axis=axis))


def softmax_backward(grad_out: np.ndarray, probs: np.ndarray, axis: int = -1) -> np.ndarray:
    r"""Vector-Jacobian product for softmax: :math:`p \odot (g - \langle g, p\rangle)`.

    Never materialises the (n, n) Jacobian, which is what makes attention
    backward tractable at long sequence length.
    """
    return probs * (grad_out - np.sum(grad_out * probs, axis=axis, keepdims=True))


def cross_entropy_from_logits(
    logits: np.ndarray, targets: np.ndarray, reduction: str = "mean"
) -> float | np.ndarray:
    """Softmax cross-entropy that never materialises probabilities.

    Args:
        logits: `(batch, classes)`.
        targets: `(batch,)` integer labels.
    """
    lp = log_softmax(logits, axis=-1)
    picked = -lp[np.arange(targets.shape[0]), targets]
    if reduction == "mean":
        return float(picked.mean())
    if reduction == "sum":
        return float(picked.sum())
    return picked


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Branchless stable logistic: `exp` never sees a large positive argument."""
    pos = x >= 0
    z = np.exp(-np.abs(x))
    return np.where(pos, 1.0 / (1.0 + z), z / (1.0 + z))


def naive_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gelu(x: np.ndarray, approximate: bool = False) -> np.ndarray:
    """Gaussian Error Linear Unit.

    Args:
        approximate: use the tanh approximation (what GPT-2 shipped, and still
            the default in several inference kernels). Max absolute deviation
            from the exact form is ~1e-3 - measured in the stability report.
    """
    if approximate:
        inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + np.tanh(inner))
    from math import sqrt

    from scipy.special import erf  # local import: only the exact path needs SciPy

    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish: `x * sigmoid(x)`."""
    return x * sigmoid(x)


def swiglu(x: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """SwiGLU: `silu(gate) * x`, the FFN activation used by Llama/Qwen-family models.

    A SwiGLU FFN has three weight matrices instead of two, so hidden sizes are
    conventionally scaled by 2/3 to keep the parameter count comparable.
    """
    return silu(gate) * x
