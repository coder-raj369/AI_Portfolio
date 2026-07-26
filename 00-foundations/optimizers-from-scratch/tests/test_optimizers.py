"""Update rules are checked against closed-form expectations, then against
actual convergence on an ill-conditioned quadratic."""

from __future__ import annotations

import numpy as np
import pytest

from nanoptim import SGD, Adam, AdamW, Lion, Parameter, clip_grad_norm_


def _quadratic(condition: float = 100.0, dim: int = 20, seed: int = 0):
    """f(x) = 0.5 x^T A x with eigenvalues log-spaced over [1, condition]."""
    rng = np.random.default_rng(seed)
    eig = np.logspace(0, np.log10(condition), dim)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    A = q @ np.diag(eig) @ q.T
    x0 = rng.standard_normal(dim) * 2.0

    def loss_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(0.5 * x @ A @ x), A @ x

    return loss_and_grad, x0


OPTIMISERS = {
    "sgd": lambda p: SGD(p, lr=3e-3),
    "sgd_momentum": lambda p: SGD(p, lr=1e-3, momentum=0.9),
    "sgd_nesterov": lambda p: SGD(p, lr=1e-3, momentum=0.9, nesterov=True),
    "adam": lambda p: Adam(p, lr=0.1),
    "adamw": lambda p: AdamW(p, lr=0.1, weight_decay=0.0),
    "lion": lambda p: Lion(p, lr=0.02),
}


@pytest.mark.parametrize("name", sorted(OPTIMISERS))
def test_optimiser_minimises_ill_conditioned_quadratic(name: str) -> None:
    loss_and_grad, x0 = _quadratic()
    param = Parameter(x0)
    opt = OPTIMISERS[name]([param])
    first, _ = loss_and_grad(param.data)
    for _ in range(2000):
        _, grad = loss_and_grad(param.data)
        param.grad = grad
        opt.step()
    final, _ = loss_and_grad(param.data)
    assert final < first * 1e-3, f"{name}: {first:.3e} -> {final:.3e}"
    assert np.all(np.isfinite(param.data))


def test_sgd_plain_step_is_exactly_lr_times_grad() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    opt = SGD([p], lr=0.1)
    p.grad = np.array([1.0, -3.0])
    opt.step()
    np.testing.assert_allclose(p.data, [0.9, 2.3])


def test_sgd_momentum_matches_closed_form() -> None:
    """v1 = g, v2 = mu*g + g  =>  total displacement = lr*(2 + mu)*g."""
    p = Parameter(np.array([0.0]))
    opt = SGD([p], lr=0.1, momentum=0.9)
    for _ in range(2):
        p.grad = np.array([1.0])
        opt.step()
    np.testing.assert_allclose(p.data, [-0.1 * (2 + 0.9)], rtol=1e-12)


def test_nesterov_takes_a_larger_first_effective_step() -> None:
    plain, nest = Parameter(np.zeros(1)), Parameter(np.zeros(1))
    o1, o2 = SGD([plain], lr=0.1, momentum=0.9), SGD([nest], lr=0.1, momentum=0.9, nesterov=True)
    for _ in range(3):
        plain.grad = np.ones(1)
        nest.grad = np.ones(1)
        o1.step()
        o2.step()
    assert abs(nest.data.item()) > abs(plain.data.item())


def test_adam_first_step_is_approximately_lr_times_sign() -> None:
    """Bias correction makes step 1 scale-free: |dtheta| ~= lr regardless of |g|."""
    for scale in (1e-3, 1.0, 1e3):
        p = Parameter(np.array([0.0]))
        opt = Adam([p], lr=0.01)
        p.grad = np.array([scale])
        opt.step()
        assert p.data.item() == pytest.approx(-0.01, rel=1e-4)


def test_bias_correction_prevents_an_oversized_first_step() -> None:
    """Without bias correction, step 1 would be lr*(1-b1)/sqrt(1-b2) = 3.16x too big.

    m1 = 0.1g and v1 = 0.001g^2, so the raw ratio m/sqrt(v) is 3.16 rather than 1.
    Dropping the correction terms is therefore not a harmless simplification: it
    front-loads an over-large step exactly when the variance estimate is worst.
    """
    p = Parameter(np.array([0.0]))
    opt = Adam([p], lr=0.01)
    p.grad = np.array([1.0])
    opt.step()
    corrected = abs(p.data.item())
    uncorrected = 0.01 * (1 - 0.9) / np.sqrt(1 - 0.999)
    assert corrected == pytest.approx(0.01, rel=1e-4)
    assert uncorrected / corrected == pytest.approx(3.162, rel=1e-2)


def test_lion_update_has_unit_magnitude_per_coordinate() -> None:
    p = Parameter(np.array([0.0, 0.0, 0.0]))
    opt = Lion([p], lr=0.05)
    p.grad = np.array([1e-6, -400.0, 7.0])  # wildly different gradient scales
    opt.step()
    np.testing.assert_allclose(np.abs(p.data), [0.05, 0.05, 0.05], rtol=1e-12)


def test_lion_keeps_one_state_buffer_where_adam_keeps_two() -> None:
    p = Parameter(np.zeros(4))
    lion, adam = Lion([p], lr=0.01), Adam([Parameter(np.zeros(4))], lr=0.01)
    p.grad = np.ones(4)
    lion.step()
    adam.params[0].grad = np.ones(4)
    adam.step()
    assert len(lion.m) == 1
    assert len(adam.m) + len(adam.v) == 2


def test_zero_grad_and_missing_grad_are_no_ops() -> None:
    p = Parameter(np.array([1.0]))
    opt = Adam([p], lr=0.1)
    opt.step()  # grad is None: must not raise, must not move the parameter
    np.testing.assert_allclose(p.data, [1.0])
    p.grad = np.array([1.0])
    opt.zero_grad()
    opt.step()
    np.testing.assert_allclose(p.data, [1.0])


def test_clip_grad_norm_scales_to_target_and_reports_pre_clip_norm() -> None:
    a, b = Parameter(np.zeros(2)), Parameter(np.zeros(2))
    a.grad, b.grad = np.array([3.0, 0.0]), np.array([0.0, 4.0])
    pre = clip_grad_norm_([a, b], max_norm=1.0)
    assert pre == pytest.approx(5.0)
    post = np.sqrt((a.grad**2).sum() + (b.grad**2).sum())
    assert post == pytest.approx(1.0, rel=1e-6)


def test_clip_grad_norm_leaves_small_gradients_untouched() -> None:
    a = Parameter(np.zeros(2))
    a.grad = np.array([0.1, 0.1])
    before = a.grad.copy()
    clip_grad_norm_([a], max_norm=10.0)
    np.testing.assert_array_equal(a.grad, before)


def test_invalid_hyperparameters_raise() -> None:
    p = Parameter(np.zeros(1))
    with pytest.raises(ValueError):
        SGD([p], lr=-1.0)
    with pytest.raises(ValueError):
        SGD([p], lr=0.1, nesterov=True, momentum=0.0)
    with pytest.raises(ValueError):
        Adam([p], lr=0.1, betas=(1.0, 0.999))
