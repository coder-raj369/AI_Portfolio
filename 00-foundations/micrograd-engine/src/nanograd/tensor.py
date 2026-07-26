"""The autodiff core: a `Tensor` that records a backward closure per operation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

ArrayLike = Any
Shape = tuple[int, ...]


def _unbroadcast(grad: np.ndarray, shape: Shape) -> np.ndarray:
    """Reduce `grad` back to `shape`, undoing NumPy broadcasting.

    This is the single most common source of silent bugs in hand-rolled autodiff:
    a forward pass broadcasts (4, 1) against (4, 8), and the backward pass must
    sum the gradient over the axes that were expanded instead of returning a
    mis-shaped array that NumPy will then happily broadcast again.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def _expand_for_axis(grad: np.ndarray, shape: Shape, axis: int | tuple[int, ...] | None,
                     keepdims: bool) -> np.ndarray:
    """Broadcast a reduced gradient back over the axes that were reduced away."""
    if axis is None:
        return np.broadcast_to(grad, shape).copy()
    axes = (axis,) if isinstance(axis, int) else axis
    if not keepdims:
        grad = np.expand_dims(grad, [a % len(shape) for a in axes])
    return np.broadcast_to(grad, shape).copy()


class Tensor:
    """A node in the computation graph.

    Attributes:
        data: the forward value.
        grad: accumulated gradient of the scalar output w.r.t. this tensor,
            or `None` until `backward()` has run.
        requires_grad: whether gradients should flow to this node.
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Iterable[Tensor] = (),
        _op: str = "leaf",
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self._prev: tuple[Tensor, ...] = tuple(_children)
        self.requires_grad = bool(requires_grad) or any(c.requires_grad for c in self._prev)
        self._op = _op
        self._backward = lambda: None

    # ---------------------------------------------------------------- helpers
    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op}, requires_grad={self.requires_grad})"

    def item(self) -> float:
        return float(self.data.reshape(()))

    def detach(self) -> Tensor:
        """Return a value-identical tensor cut out of the graph."""
        return Tensor(self.data.copy())

    def zero_grad(self) -> None:
        self.grad = None

    def _accumulate(self, grad: np.ndarray) -> None:
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    @staticmethod
    def _wrap(other: ArrayLike | Tensor) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(other)

    def _make(self, data: np.ndarray, children: tuple[Tensor, ...], op: str) -> Tensor:
        return Tensor(data, _children=children, _op=op)

    # ------------------------------------------------------------ binary ops
    def __add__(self, other: ArrayLike | Tensor) -> Tensor:
        other = self._wrap(other)
        out = self._make(self.data + other.data, (self, other), "add")

        def _backward() -> None:
            g = out.grad
            if self.requires_grad:
                self._accumulate(_unbroadcast(g, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(g, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike | Tensor) -> Tensor:
        other = self._wrap(other)
        out = self._make(self.data * other.data, (self, other), "mul")

        def _backward() -> None:
            g = out.grad
            if self.requires_grad:
                self._accumulate(_unbroadcast(g * other.data, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(g * self.data, other.shape))

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        other = self._wrap(other)
        out = self._make(self.data @ other.data, (self, other), "matmul")

        def _backward() -> None:
            g = out.grad
            if self.requires_grad:
                self._accumulate(_unbroadcast(g @ np.swapaxes(other.data, -1, -2), self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, other.shape))

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Tensor:
        if isinstance(exponent, Tensor):
            raise TypeError("tensor**tensor is not supported; use (x.log() * y).exp()")
        out = self._make(self.data**exponent, (self,), f"pow{exponent}")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * exponent * self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: ArrayLike | Tensor) -> Tensor:
        return self + (-self._wrap(other))

    def __truediv__(self, other: ArrayLike | Tensor) -> Tensor:
        return self * (self._wrap(other) ** -1.0)

    def __radd__(self, other: ArrayLike) -> Tensor:
        return self + other

    def __rmul__(self, other: ArrayLike) -> Tensor:
        return self * other

    def __rsub__(self, other: ArrayLike) -> Tensor:
        return self._wrap(other) + (-self)

    def __rtruediv__(self, other: ArrayLike) -> Tensor:
        return self._wrap(other) * (self**-1.0)

    # ------------------------------------------------------------- unary ops
    def exp(self) -> Tensor:
        out = self._make(np.exp(self.data), (self,), "exp")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = self._make(np.log(self.data), (self,), "log")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad / self.data)

        out._backward = _backward
        return out

    def sqrt(self) -> Tensor:
        return self**0.5

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        out = self._make(t, (self,), "tanh")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * (1.0 - t * t))

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = self._make(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward() -> None:
            if self.requires_grad:
                # Subgradient at exactly 0 is taken as 0 (the PyTorch convention).
                self._accumulate(out.grad * (self.data > 0.0))

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        # Branchless stable logistic: exp() never sees a large positive argument.
        pos = self.data >= 0
        z = np.exp(-np.abs(self.data))
        s = np.where(pos, 1.0 / (1.0 + z), z / (1.0 + z))
        out = self._make(s, (self,), "sigmoid")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * s * (1.0 - s))

        out._backward = _backward
        return out

    # ------------------------------------------------------------ reductions
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(_expand_for_axis(out.grad, self.shape, axis, keepdims))

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        n = self.data.size // max(np.asarray(self.data.sum(axis=axis, keepdims=True)).size, 1)
        out = self._make(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(_expand_for_axis(out.grad, self.shape, axis, keepdims) / n)

        out._backward = _backward
        return out

    def max(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        m = self.data.max(axis=axis, keepdims=True)
        out = self._make(
            m if keepdims else self.data.max(axis=axis, keepdims=False), (self,), "max"
        )

        def _backward() -> None:
            if self.requires_grad:
                mask = (self.data == m).astype(np.float64)
                # Ties split the gradient evenly, which keeps gradcheck honest.
                mask /= mask.sum(axis=axis, keepdims=True)
                self._accumulate(_expand_for_axis(out.grad, self.shape, axis, keepdims) * mask)

        out._backward = _backward
        return out

    # ------------------------------------------------------------- structural
    def reshape(self, *shape: int) -> Tensor:
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        out = self._make(self.data.reshape(target), (self,), "reshape")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        out = self._make(np.transpose(self.data, axes), (self,), "transpose")

        def _backward() -> None:
            if self.requires_grad:
                inv = None if axes is None else tuple(np.argsort(axes))
                self._accumulate(np.transpose(out.grad, inv))

        out._backward = _backward
        return out

    def __getitem__(self, index: Any) -> Tensor:
        out = self._make(self.data[index], (self,), "getitem")

        def _backward() -> None:
            if self.requires_grad:
                g = np.zeros_like(self.data)
                np.add.at(g, index, out.grad)  # add.at, not `+=`: repeated indices must accumulate
                self._accumulate(g)

        out._backward = _backward
        return out

    # --------------------------------------------------------------- backward
    def _topological_order(self) -> list[Tensor]:
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:  # iterative DFS: recursion would blow the stack on long graphs
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in seen:
                    stack.append((child, False))
        return order

    def backward(self, grad: ArrayLike | None = None) -> None:
        """Backpropagate from this tensor.

        Args:
            grad: seed gradient. Defaults to ones, which is only meaningful for a
                scalar output - calling `backward()` on a non-scalar without an
                explicit seed is a bug, so it raises.
        """
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() on a non-scalar tensor requires an explicit `grad` seed"
                )
            grad = np.ones_like(self.data)
        self.grad = np.asarray(grad, dtype=np.float64).reshape(self.shape).copy()
        for node in reversed(self._topological_order()):
            node._backward()
