"""Graph-mechanics behaviour that gradcheck alone would not catch."""

from __future__ import annotations

import numpy as np
import pytest

from nanograd import Tensor


def test_diamond_graph_accumulates_both_paths() -> None:
    """x is used twice; the two contributions must add, not overwrite."""
    x = Tensor(3.0, requires_grad=True)
    y = x * x + x  # dy/dx = 2x + 1 = 7
    y.backward()
    assert float(x.grad) == pytest.approx(7.0)


def test_reused_subgraph_is_visited_once() -> None:
    x = Tensor(2.0, requires_grad=True)
    shared = x * 5.0
    out = shared + shared  # d/dx = 10
    out.backward()
    assert float(x.grad) == pytest.approx(10.0)


def test_backward_on_non_scalar_requires_seed() -> None:
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    with pytest.raises(RuntimeError, match="non-scalar"):
        (x * 2.0).backward()


def test_explicit_seed_is_respected() -> None:
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    out = x * 3.0
    out.backward(np.full((2, 2), 2.0))
    np.testing.assert_allclose(x.grad, np.full((2, 2), 6.0))


def test_requires_grad_propagates_and_blocks() -> None:
    a = Tensor(np.ones(3), requires_grad=True)
    b = Tensor(np.ones(3))
    assert (a * b).requires_grad
    assert not (b * b).requires_grad
    (b * b).backward(np.ones(3))
    assert b.grad is None  # no gradient stored for a non-differentiable leaf


def test_detach_cuts_the_graph() -> None:
    a = Tensor(np.ones(3), requires_grad=True)
    d = (a * 2.0).detach()
    assert not d.requires_grad
    (d * 5.0).backward(np.ones(3))
    assert a.grad is None


def test_zero_grad_clears_accumulation() -> None:
    x = Tensor(2.0, requires_grad=True)
    for _ in range(3):
        (x * x).backward()
    assert float(x.grad) == pytest.approx(12.0)  # accumulated over 3 passes
    x.zero_grad()
    (x * x).backward()
    assert float(x.grad) == pytest.approx(4.0)


def test_max_splits_gradient_across_ties() -> None:
    x = Tensor(np.array([[1.0, 1.0, 0.0]]), requires_grad=True)
    x.max(axis=1).sum().backward()
    np.testing.assert_allclose(x.grad, np.array([[0.5, 0.5, 0.0]]))


def test_long_chain_does_not_recurse_to_death() -> None:
    """5k-deep graph: an implementation using recursive DFS raises RecursionError."""
    x = Tensor(1.0, requires_grad=True)
    node = x
    for _ in range(5000):
        node = node * 1.0001
    node.backward()
    assert np.isfinite(x.grad)


def test_transpose_gradient_shape() -> None:
    x = Tensor(np.random.default_rng(0).standard_normal((3, 5)), requires_grad=True)
    (x.T * 2.0).sum().backward()
    assert x.grad.shape == (3, 5)
