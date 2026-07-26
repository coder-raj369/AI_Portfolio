"""Activations and losses built on the `Tensor` primitives.

`log_softmax` and `nll_loss` are implemented as primitives rather than as
compositions of exp/log/sum because that is where numerical stability and a
clean gradient live. `softmax` then reuses `log_softmax`, so there is exactly
one place in the codebase that has to get the max-subtraction trick right.
"""

from __future__ import annotations

import numpy as np

from nanograd.tensor import Tensor


def relu(x: Tensor) -> Tensor:
    return x.relu()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable log-softmax.

    Computed as `x - m - log(sum(exp(x - m)))` with `m = max(x)`, so the largest
    argument to `exp` is 0. The naive `log(softmax(x))` overflows for logits
    around 750+ in float64 and around 90 in float32; see
    `nn-primitives/experiments/stability_report.py` for the measured failure point.
    """
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    out_data = shifted - logsumexp
    out = Tensor(out_data, _children=(x,), _op="log_softmax")

    def _backward() -> None:
        if x.requires_grad:
            g = out.grad
            probs = np.exp(out_data)
            # d/dx logsoftmax(x) . g  =  g - softmax(x) * sum(g)
            x._accumulate(g - probs * g.sum(axis=axis, keepdims=True))

    out._backward = _backward
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return log_softmax(x, axis=axis).exp()


def nll_loss(log_probs: Tensor, targets: np.ndarray, reduction: str = "mean") -> Tensor:
    """Negative log-likelihood over integer class targets.

    Args:
        log_probs: `(batch, classes)` log-probabilities (output of `log_softmax`).
        targets: `(batch,)` integer class indices.
        reduction: `"mean"` or `"sum"`.
    """
    targets = np.asarray(targets, dtype=np.int64)
    if log_probs.ndim != 2:
        raise ValueError(f"expected (batch, classes) log-probs, got {log_probs.shape}")
    if targets.shape != (log_probs.shape[0],):
        raise ValueError(f"targets {targets.shape} do not match batch {log_probs.shape[0]}")
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"unknown reduction {reduction!r}")

    rows = np.arange(targets.shape[0])
    picked = log_probs.data[rows, targets]
    total = -picked.sum()
    value = total / targets.shape[0] if reduction == "mean" else total
    out = Tensor(value, _children=(log_probs,), _op=f"nll_{reduction}")

    def _backward() -> None:
        if log_probs.requires_grad:
            scale = 1.0 / targets.shape[0] if reduction == "mean" else 1.0
            g = np.zeros_like(log_probs.data)
            g[rows, targets] = -scale * float(out.grad)
            log_probs._accumulate(g)

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray, reduction: str = "mean") -> Tensor:
    """Softmax cross-entropy from logits (the stable path: never materialises probs)."""
    return nll_loss(log_softmax(logits, axis=-1), targets, reduction=reduction)


def mse_loss(pred: Tensor, target: Tensor | np.ndarray, reduction: str = "mean") -> Tensor:
    diff = pred - Tensor._wrap(target)
    sq = diff * diff
    return sq.mean() if reduction == "mean" else sq.sum()
