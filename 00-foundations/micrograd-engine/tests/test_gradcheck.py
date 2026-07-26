"""Every op's analytic gradient is checked against central finite differences."""

from __future__ import annotations

import numpy as np
import pytest

from nanograd import Tensor, cross_entropy, gradcheck, log_softmax, mse_loss, softmax

RNG = np.random.default_rng(7)


def _t(*shape: int, positive: bool = False) -> Tensor:
    data = RNG.standard_normal(shape)
    if positive:
        data = np.abs(data) + 0.5  # keep log/pow domains valid
    return Tensor(data, requires_grad=True)


CASES = {
    "add": lambda a, b: (a + b).sum(),
    "sub": lambda a, b: (a - b).sum(),
    "mul": lambda a, b: (a * b).sum(),
    "div": lambda a, b: (a / (b * b + 1.0)).sum(),
    "pow3": lambda a, b: (a**3).sum() + (b**2).sum(),
    "exp": lambda a, b: (a * 0.3).exp().sum() + b.sum(),
    "tanh": lambda a, b: (a @ b.transpose()).tanh().sum() if a.shape[1] == b.shape[1] else None,
    "relu": lambda a, b: (a + b).relu().sum(),
    "sigmoid": lambda a, b: (a * b).sigmoid().sum(),
    "mean": lambda a, b: (a * b).mean(),
    "max_axis": lambda a, b: (a + b).max(axis=1).sum(),
    "sum_axis_keepdims": lambda a, b: ((a + b).sum(axis=0, keepdims=True) * 2.0).sum(),
    "reshape": lambda a, b: (a + b).reshape(-1).sum(),
    "chain": lambda a, b: ((a * b).tanh() + (a - b).relu()).mean(),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_op_gradients_match_finite_differences(name: str) -> None:
    a, b = _t(4, 5), _t(4, 5)
    fn = CASES[name]
    if fn(a, b) is None:
        pytest.skip("shape-incompatible case")
    err = gradcheck(lambda: fn(a, b), [a, b], tol=1e-5)
    assert err < 1e-5, f"{name}: relative error {err:.2e}"


def test_log_and_sqrt_gradients() -> None:
    a = _t(3, 4, positive=True)
    assert gradcheck(lambda: (a.log() + a.sqrt()).sum(), [a]) < 1e-5


def test_matmul_gradients_with_broadcasting_bias() -> None:
    x = _t(6, 3)
    w = _t(3, 4)
    b = Tensor(RNG.standard_normal((1, 4)), requires_grad=True)  # broadcast over batch
    assert gradcheck(lambda: ((x @ w) + b).tanh().sum(), [x, w, b]) < 1e-5


def test_cross_entropy_gradient() -> None:
    logits = _t(8, 5)
    targets = RNG.integers(0, 5, size=8)
    assert gradcheck(lambda: cross_entropy(logits, targets), [logits]) < 1e-5


def test_softmax_rows_sum_to_one_and_gradient_is_correct() -> None:
    logits = _t(4, 6)
    probs = softmax(logits)
    np.testing.assert_allclose(probs.data.sum(axis=-1), np.ones(4), rtol=0, atol=1e-12)
    # A non-trivial scalar function of the softmax, so the Jacobian is exercised.
    weights = RNG.standard_normal((4, 6))
    assert gradcheck(lambda: (softmax(logits) * Tensor(weights)).sum(), [logits]) < 1e-5


def test_mse_gradient() -> None:
    pred, target = _t(5, 2), _t(5, 2)
    assert gradcheck(lambda: mse_loss(pred, target), [pred, target]) < 1e-5


def test_log_softmax_matches_naive_formula_in_safe_range() -> None:
    x = Tensor(RNG.standard_normal((3, 7)))
    naive = np.log(np.exp(x.data) / np.exp(x.data).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(log_softmax(x).data, naive, rtol=1e-12, atol=1e-12)


def test_getitem_gradient_with_repeated_indices() -> None:
    """Repeated indices must accumulate; `g[idx] += v` silently drops duplicates."""
    x = Tensor(np.arange(5.0), requires_grad=True)
    idx = np.array([1, 1, 1, 3])
    out = x[idx].sum()
    out.backward()
    np.testing.assert_array_equal(x.grad, np.array([0.0, 3.0, 0.0, 1.0, 0.0]))
