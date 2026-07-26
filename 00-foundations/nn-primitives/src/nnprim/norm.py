r"""LayerNorm and RMSNorm, forward and hand-derived backward.

The backward pass for LayerNorm is a standard interview question and a standard
place to be quietly wrong, because :math:`\mu` and :math:`\sigma` both depend on
every element of the input - so the gradient has three terms, not one.
"""

from __future__ import annotations

import numpy as np


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    eps: float = 1e-5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    r""":math:`y = \gamma \cdot (x - \mu) / \sqrt{\sigma^2 + \epsilon} + \beta`.

    Returns:
        `(y, cache)` where `cache` holds what the backward pass needs.
    """
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = xc * inv
    y = xhat if gamma is None else xhat * gamma
    if beta is not None:
        y = y + beta
    return y, {"xhat": xhat, "inv": inv, "gamma": gamma}


def layer_norm_backward(
    grad_y: np.ndarray, cache: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    r"""Backward pass for `layer_norm`.

    .. math::
        \frac{\partial L}{\partial x} = \frac{1}{\sigma}\left(g
        - \overline{g} - \hat{x}\,\overline{g \hat{x}}\right)

    where :math:`g = \partial L/\partial y \cdot \gamma` and the bars are means
    over the normalised axis. The second and third terms are the ones that get
    dropped by accident; they are what keep the gradient orthogonal to the
    directions LayerNorm has already removed (mean and scale).
    """
    xhat, inv, gamma = cache["xhat"], cache["inv"], cache["gamma"]
    d = xhat.shape[-1]
    grad_gamma = (grad_y * xhat).reshape(-1, d).sum(axis=0) if gamma is not None else None
    grad_beta = grad_y.reshape(-1, d).sum(axis=0)
    g = grad_y if gamma is None else grad_y * gamma
    grad_x = inv * (
        g
        - g.mean(axis=-1, keepdims=True)
        - xhat * (g * xhat).mean(axis=-1, keepdims=True)
    )
    return grad_x, grad_gamma, grad_beta


def rms_norm(
    x: np.ndarray, gamma: np.ndarray | None = None, eps: float = 1e-6
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    r""":math:`y = \gamma \cdot x / \sqrt{\mathrm{mean}(x^2) + \epsilon}`.

    Drops the mean subtraction and the bias entirely. That removes one pass over
    the row and one buffer, which is why every recent decoder-only LLM
    (Llama, Qwen, Gemma, Mistral) uses it: same quality, cheaper kernel, and one
    less quantity to keep in fp32 during mixed-precision training.
    """
    ms = np.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    xhat = x * inv
    return (xhat if gamma is None else xhat * gamma), {"x": x, "inv": inv, "gamma": gamma}


def rms_norm_backward(
    grad_y: np.ndarray, cache: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Backward pass for `rms_norm`.

    .. math::
        \frac{\partial L}{\partial x} = s\left(g - x\,\frac{\langle g, x
        \rangle}{d\,\mathrm{ms}}\right), \qquad s = 1/\sqrt{\mathrm{ms}+\epsilon}
    """
    x, inv, gamma = cache["x"], cache["inv"], cache["gamma"]
    d = x.shape[-1]
    grad_gamma = (grad_y * (x * inv)).reshape(-1, d).sum(axis=0) if gamma is not None else None
    g = grad_y if gamma is None else grad_y * gamma
    grad_x = inv * (g - x * inv * inv * np.mean(g * x, axis=-1, keepdims=True))
    return grad_x, grad_gamma
