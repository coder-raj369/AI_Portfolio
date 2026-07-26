"""Minimal module system: enough to build and train a real MLP, nothing more."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from nanograd.functional import relu, tanh
from nanograd.tensor import Tensor


class Module:
    """Base class: tracks child modules/parameters by attribute assignment."""

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for value in vars(self).values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(int(p.data.size) for p in self.parameters())

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class Linear(Module):
    """`y = x @ W + b` with Kaiming-uniform init (the PyTorch default scheme)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(0)
        bound = 1.0 / np.sqrt(in_features)
        self.weight = Tensor(
            rng.uniform(-bound, bound, size=(in_features, out_features)), requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)


class MLP(Sequential):
    """Fully-connected net: `sizes=[in, h1, ..., out]`, activation between layers."""

    def __init__(self, sizes: Sequence[int], activation: str = "relu", seed: int = 0) -> None:
        if len(sizes) < 2:
            raise ValueError("need at least an input and an output size")
        act = {"relu": ReLU, "tanh": Tanh}[activation]
        rng = np.random.default_rng(seed)
        layers: list[Module] = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(Linear(a, b, rng=rng))
            if i < len(sizes) - 2:
                layers.append(act())
        super().__init__(*layers)
