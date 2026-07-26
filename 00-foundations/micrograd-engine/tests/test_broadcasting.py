"""Broadcasting is where hand-rolled autodiff usually breaks."""

from __future__ import annotations

import numpy as np
import pytest

from nanograd import Tensor, gradcheck

RNG = np.random.default_rng(11)

SHAPE_PAIRS = [
    ((4, 5), (5,)),
    ((4, 5), (1, 5)),
    ((4, 5), (4, 1)),
    ((3, 1, 5), (1, 4, 5)),
    ((2, 3, 4), (4,)),
    ((6, 1), (1, 6)),
]


@pytest.mark.parametrize("shape_a,shape_b", SHAPE_PAIRS)
def test_broadcast_gradients_keep_input_shapes(shape_a, shape_b) -> None:
    a = Tensor(RNG.standard_normal(shape_a), requires_grad=True)
    b = Tensor(RNG.standard_normal(shape_b), requires_grad=True)
    out = (a * b).sum()
    out.backward()
    assert a.grad.shape == shape_a
    assert b.grad.shape == shape_b


@pytest.mark.parametrize("shape_a,shape_b", SHAPE_PAIRS)
def test_broadcast_gradients_are_numerically_right(shape_a, shape_b) -> None:
    a = Tensor(RNG.standard_normal(shape_a), requires_grad=True)
    b = Tensor(RNG.standard_normal(shape_b), requires_grad=True)
    assert gradcheck(lambda: ((a + b) * (a * b)).sum(), [a, b]) < 1e-5


def test_batched_matmul_broadcasts_over_leading_dims() -> None:
    a = Tensor(RNG.standard_normal((2, 3, 4)), requires_grad=True)
    b = Tensor(RNG.standard_normal((4, 5)), requires_grad=True)
    out = (a @ b).sum()
    out.backward()
    assert a.grad.shape == (2, 3, 4)
    assert b.grad.shape == (4, 5)
    assert gradcheck(lambda: (a @ b).tanh().sum(), [a, b]) < 1e-5
