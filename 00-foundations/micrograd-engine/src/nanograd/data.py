"""Small synthetic datasets that are actually hard enough to be informative.

A two-layer net gets ~100% on linearly separable blobs, which proves nothing.
The interleaved-spirals task needs real non-linearity, so a broken gradient
shows up immediately as a stalled decision boundary.
"""

from __future__ import annotations

import numpy as np


def make_spirals(
    n_per_class: int = 400,
    n_classes: int = 3,
    noise: float = 0.08,
    turns: float = 1.35,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate `n_classes` interleaved Archimedean spirals.

    Returns:
        `X` of shape `(n_per_class * n_classes, 2)` and integer `y`.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        t = np.linspace(0.15, 1.0, n_per_class) ** 0.65 * (2 * np.pi * turns)
        r = t / (2 * np.pi * turns)
        theta = t + c * (2 * np.pi / n_classes)
        pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        pts += rng.normal(scale=noise * r[:, None], size=pts.shape)
        xs.append(pts)
        ys.append(np.full(n_per_class, c))
    X = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.int64)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_frac: float = 0.25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    cut = int(round(X.shape[0] * (1 - test_frac)))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]
