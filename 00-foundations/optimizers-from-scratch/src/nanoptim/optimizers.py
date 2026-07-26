"""SGD (+momentum/Nesterov), Adam, AdamW and Lion, written out step by step.

The interesting part of this module is not that the update rules exist - it is
that `Adam(weight_decay=...)` and `AdamW(weight_decay=...)` are *not* the same
algorithm, and the tests prove it numerically rather than citing the paper.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Param(Protocol):
    """Anything with a value and a gradient: `nanograd.Tensor` qualifies."""

    data: np.ndarray
    grad: np.ndarray | None


class Parameter:
    """A plain-NumPy parameter, for optimiser benchmarks that need no autodiff."""

    __slots__ = ("data", "grad")

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.array(data, dtype=np.float64)
        self.grad: np.ndarray | None = None

    def zero_grad(self) -> None:
        self.grad = None


def clip_grad_norm_(params: Iterable[Param], max_norm: float) -> float:
    """Scale gradients in place so their global L2 norm is at most `max_norm`.

    Returns the pre-clip norm, which is the quantity worth logging: a spiking
    grad-norm is the earliest visible symptom of a diverging run.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
    if total > max_norm > 0:
        scale = max_norm / (total + 1e-12)
        for g in grads:
            g *= scale
    return total


class Optimizer:
    """Base class holding the parameter list, step counter and LR bookkeeping."""

    def __init__(self, params: Sequence[Param], lr: float) -> None:
        self.params = list(params)
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.lr = float(lr)
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    def step(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _iter_grads(self) -> Iterable[tuple[int, Param, np.ndarray]]:
        for i, p in enumerate(self.params):
            if p.grad is not None:
                yield i, p, p.grad


class SGD(Optimizer):
    r"""Stochastic gradient descent with optional momentum / Nesterov / L2.

    Heavy-ball momentum, in the PyTorch parameterisation:

    .. math::
        v_t = \mu v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta v_t

    Note this is *not* the "exponential average" form `v = mu*v + (1-mu)*g`;
    the effective step is larger by `1/(1-mu)`, which is why lr=0.1 with mu=0.9
    behaves like lr=1.0 without momentum.
    """

    def __init__(
        self,
        params: Sequence[Param],
        lr: float = 1e-2,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.buf: dict[int, np.ndarray] = {}

    def step(self) -> None:
        self.t += 1
        for i, p, grad in self._iter_grads():
            g = grad + self.weight_decay * p.data if self.weight_decay else grad
            if self.momentum:
                buf = self.buf.get(i)
                buf = g.copy() if buf is None else self.momentum * buf + g
                self.buf[i] = buf
                g = g + self.momentum * buf if self.nesterov else buf
            p.data -= self.lr * g


class Adam(Optimizer):
    r"""Adam with bias correction and *coupled* L2 regularisation.

    .. math::
        m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
        v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2

    `weight_decay` here is added into the gradient (`g += wd * theta`), i.e. the
    original Adam-with-L2. That decay term is then divided by
    :math:`\sqrt{\hat v}` along with everything else, so its effective strength
    depends on the gradient scale of each parameter - the pathology AdamW fixes.
    """

    def __init__(
        self,
        params: Sequence[Param],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}
        self.decoupled = False

    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        for i, p, grad in self._iter_grads():
            g = grad
            if self.weight_decay and not self.decoupled:
                g = g + self.weight_decay * p.data
            m = self.m.get(i)
            v = self.v.get(i)
            m = np.zeros_like(p.data) if m is None else m
            v = np.zeros_like(p.data) if v is None else v
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            self.m[i], self.v[i] = m, v
            update = (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            if self.weight_decay and self.decoupled:
                # AdamW: decay applied straight to the weights, outside the
                # adaptive denominator, and scaled only by the learning rate.
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * update


class AdamW(Adam):
    """Adam with *decoupled* weight decay (Loshchilov & Hutter, 2019).

    Identical to `Adam` except the decay term never touches the moment
    estimates. `tests/test_adamw_decoupling.py` shows the two disagree by ~3
    orders of magnitude on a parameter whose gradient is large.
    """

    def __init__(
        self,
        params: Sequence[Param],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay)
        self.decoupled = True


class Lion(Optimizer):
    r"""EvoLved Sign Momentum (Chen et al., 2023).

    .. math::
        c_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
        \theta_t = \theta_{t-1} - \eta\,(\mathrm{sign}(c_t) + \lambda \theta_{t-1}), \\
        m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t

    Two properties make Lion worth implementing next to Adam: the update has
    **unit magnitude per coordinate** (so the effective step is exactly `lr`,
    which is why Lion needs a ~3-10x smaller lr than Adam), and it keeps one
    momentum buffer instead of two - a 33% optimiser-state memory saving that
    starts to matter at billion-parameter scale.
    """

    def __init__(
        self,
        params: Sequence[Param],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.m: dict[int, np.ndarray] = {}

    def step(self) -> None:
        self.t += 1
        for i, p, grad in self._iter_grads():
            m = self.m.get(i)
            m = np.zeros_like(p.data) if m is None else m
            update = np.sign(self.beta1 * m + (1.0 - self.beta1) * grad)
            if self.weight_decay:
                update = update + self.weight_decay * p.data
            p.data -= self.lr * update
            self.m[i] = self.beta2 * m + (1.0 - self.beta2) * grad
