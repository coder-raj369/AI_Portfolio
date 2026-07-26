"""End-to-end: the engine has to actually learn something non-linear.

A gradient bug can still pass gradcheck on a single op and then quietly fail to
train, so this test drives a full loop and asserts on accuracy, not on shapes.
"""

from __future__ import annotations

import numpy as np

from nanograd import MLP, Tensor, cross_entropy, make_spirals, train_test_split


def _standardise(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mu, sd = train.mean(axis=0), train.std(axis=0)
    return [(a - mu) / sd for a in (train, *others)]


def _accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    return float((model(Tensor(X)).data.argmax(axis=1) == y).mean())


def test_mlp_learns_spirals() -> None:
    X, y = make_spirals(n_per_class=180, seed=1)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=1)
    Xtr, Xte = _standardise(Xtr, Xte)

    model = MLP([2, 32, 32, 3], activation="tanh", seed=1)
    params = model.parameters()
    velocity = [np.zeros_like(p.data) for p in params]

    rng = np.random.default_rng(0)
    first_loss = None
    for step in range(800):
        idx = rng.choice(Xtr.shape[0], size=64, replace=False)
        loss = cross_entropy(model(Tensor(Xtr[idx])), ytr[idx])
        model.zero_grad()
        loss.backward()
        for p, v in zip(params, velocity):  # SGD + momentum, written out on purpose
            v *= 0.9
            v -= 0.1 * p.grad
            p.data += v
        if step == 0:
            first_loss = loss.item()

    train_acc, test_acc = _accuracy(model, Xtr, ytr), _accuracy(model, Xte, yte)
    assert first_loss > 1.0  # started at chance level, ln(3) = 1.0986
    assert loss.item() < 0.20, f"final batch loss {loss.item():.3f}"
    assert train_acc > 0.95, f"train accuracy only {train_acc:.3f}"
    assert test_acc > 0.90, f"test accuracy only {test_acc:.3f}"


def test_parameter_count_matches_hand_calculation() -> None:
    # (2*32+32) + (32*32+32) + (32*3+3) = 96 + 1056 + 99 = 1251
    assert MLP([2, 32, 32, 3]).num_parameters() == 1251


def test_loss_decreases_on_a_fixed_batch() -> None:
    X, y = make_spirals(n_per_class=60, seed=3)
    (X,) = _standardise(X)
    model = MLP([2, 24, 3], activation="tanh", seed=3)
    losses = []
    for _ in range(500):
        loss = cross_entropy(model(Tensor(X)), y)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 1.0 * p.grad
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.4, f"{losses[0]:.3f} -> {losses[-1]:.3f}"
    assert all(np.isfinite(losses))
