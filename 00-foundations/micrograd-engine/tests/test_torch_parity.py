"""Optional cross-check against PyTorch. Skipped when torch is not installed.

This is the strongest single correctness signal in the module: identical inputs,
identical gradients to within float tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanograd import Tensor, cross_entropy


def test_gradients_match_pytorch() -> None:
    torch = pytest.importorskip("torch", reason="torch is optional for this repo's CPU CI")
    rng = np.random.default_rng(5)
    x_np = rng.standard_normal((8, 6))
    w_np = rng.standard_normal((6, 4))
    targets = rng.integers(0, 4, size=8)

    x = Tensor(x_np, requires_grad=True)
    w = Tensor(w_np, requires_grad=True)
    cross_entropy((x @ w).tanh(), targets).backward()

    xt = torch.tensor(x_np, requires_grad=True)
    wt = torch.tensor(w_np, requires_grad=True)
    torch.nn.functional.cross_entropy(
        torch.tanh(xt @ wt), torch.tensor(targets, dtype=torch.long)
    ).backward()

    np.testing.assert_allclose(x.grad, xt.grad.numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(w.grad, wt.grad.numpy(), rtol=1e-10, atol=1e-12)
