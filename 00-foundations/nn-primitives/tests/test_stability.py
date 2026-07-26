"""Where the naive formulas break, and proof that the stable ones do not."""

from __future__ import annotations

import numpy as np
import pytest

from nnprim.functional import (
    cross_entropy_from_logits,
    log_softmax,
    logsumexp,
    naive_log_softmax,
    naive_logsumexp,
    naive_sigmoid,
    naive_softmax,
    sigmoid,
    softmax,
)


@pytest.mark.parametrize("magnitude", [0.0, 10.0, 100.0, 800.0, 1e4])
def test_stable_softmax_is_finite_and_normalised_at_any_magnitude(magnitude: float) -> None:
    x = np.array([[magnitude, magnitude - 1.0, magnitude - 2.0]])
    p = softmax(x)
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(1.0)
    # softmax is shift-invariant, so the answer must equal the shifted-down case
    np.testing.assert_allclose(p, softmax(x - magnitude), rtol=1e-12, atol=1e-15)


def test_naive_softmax_actually_breaks_in_float64() -> None:
    """Documents the failure this module exists to prevent."""
    x = np.array([[800.0, 799.0, 798.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        assert not np.all(np.isfinite(naive_softmax(x)))
    assert np.all(np.isfinite(softmax(x)))


def test_naive_softmax_breaks_much_earlier_in_float32() -> None:
    x = np.array([[100.0, 99.0, 98.0]], dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        assert not np.all(np.isfinite(naive_softmax(x)))
    assert np.all(np.isfinite(softmax(x)))


def test_measured_overflow_thresholds_match_dtype_theory() -> None:
    """exp overflows at ln(max_float): ~88.7 in fp32, ~709.8 in fp64."""
    for dtype, expected in ((np.float32, 88.7), (np.float64, 709.8)):
        with np.errstate(over="ignore"):
            lo, hi = 1.0, 1000.0
            for _ in range(60):  # bisect the first magnitude that overflows
                mid = (lo + hi) / 2
                if np.isfinite(np.exp(np.array(mid, dtype=dtype))):
                    lo = mid
                else:
                    hi = mid
        assert lo == pytest.approx(expected, abs=0.2)


def test_stable_logsumexp_matches_naive_in_the_safe_range() -> None:
    x = np.random.default_rng(0).standard_normal((4, 9))
    np.testing.assert_allclose(logsumexp(x), naive_logsumexp(x), rtol=1e-12)


def test_logsumexp_handles_a_fully_masked_row() -> None:
    """A row that is entirely -1e9 (all positions masked) must not produce nan."""
    x = np.full((1, 4), -1e9)
    assert np.all(np.isfinite(logsumexp(x)))
    assert np.all(np.isfinite(softmax(x)))


def test_log_softmax_survives_where_log_of_softmax_gives_minus_inf() -> None:
    """Underflow, not overflow: p=exp(-800) rounds to 0 and log(0) = -inf."""
    x = np.array([[0.0, -800.0]])
    with np.errstate(divide="ignore"):
        assert np.isneginf(naive_log_softmax(x)).any()
    stable = log_softmax(x)
    assert np.all(np.isfinite(stable))
    assert stable[0, 1] == pytest.approx(-800.0, abs=1e-6)


def test_cross_entropy_is_finite_for_a_confidently_wrong_prediction() -> None:
    """The case that produces `nan` loss in real training runs."""
    logits = np.array([[500.0, -500.0]])
    loss = cross_entropy_from_logits(logits, np.array([1]))
    assert np.isfinite(loss)
    assert loss == pytest.approx(1000.0, rel=1e-6)


def test_stable_sigmoid_saturates_instead_of_overflowing() -> None:
    x = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    s = sigmoid(x)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(s[[0, 2, 4]], [0.0, 0.5, 1.0], atol=1e-12)
    # The naive form computes exp(1000) internally: it happens to land on the
    # right answer, but only after an overflow to inf. Under `over="raise"`
    # (i.e. any run with numpy error checking turned up) it is a hard failure.
    with pytest.raises(FloatingPointError), np.errstate(over="raise"):
        naive_sigmoid(np.array([-1000.0]))
    with np.errstate(over="raise", invalid="raise"):
        sigmoid(np.array([-1000.0, 1000.0]))  # must not raise


def test_gelu_tanh_approximation_error_is_small_but_real() -> None:
    from nnprim.functional import gelu

    x = np.linspace(-6, 6, 2001)
    err = float(np.abs(gelu(x) - gelu(x, approximate=True)).max())
    assert 1e-5 < err < 5e-3, f"unexpected approximation error {err:.2e}"


def test_softmax_backward_matches_finite_differences() -> None:
    from nnprim.functional import softmax_backward

    from nnprim.gradcheck import numeric_grad

    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 6))
    w = rng.standard_normal((3, 6))
    analytic = softmax_backward(w, softmax(x))
    numeric = numeric_grad(lambda: float((softmax(x) * w).sum()), x)
    assert np.abs(analytic - numeric).max() < 1e-8
