"""The headline correctness claim of this module: Adam+L2 != AdamW.

Both are widely described as "Adam with weight decay". They are different
algorithms, and the difference is large enough to change training outcomes -
which is why every modern LLM recipe specifies AdamW explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoptim import Adam, AdamW, Parameter


def test_decay_is_identical_when_weight_decay_is_zero() -> None:
    a, b = Parameter(np.array([1.0, -2.0])), Parameter(np.array([1.0, -2.0]))
    oa, ob = Adam([a], lr=0.1, weight_decay=0.0), AdamW([b], lr=0.1, weight_decay=0.0)
    for _ in range(10):
        a.grad = np.array([0.5, 0.5])
        b.grad = np.array([0.5, 0.5])
        oa.step()
        ob.step()
    np.testing.assert_allclose(a.data, b.data, rtol=1e-12)


def test_adamw_decay_is_exactly_lr_times_wd_times_theta_when_grad_is_zero() -> None:
    """With no gradient signal, AdamW is pure exponential shrinkage: (1 - lr*wd)^t."""
    p = Parameter(np.array([1.0]))
    opt = AdamW([p], lr=0.1, weight_decay=0.1)
    for _ in range(5):
        p.grad = np.zeros(1)
        opt.step()
    assert p.data.item() == pytest.approx((1 - 0.1 * 0.1) ** 5, rel=1e-12)


def test_coupled_adam_decay_is_amplified_by_the_adaptive_denominator() -> None:
    """Same setup as above under Adam+L2: the decay term is divided by
    sqrt(v_hat), so a parameter with zero gradient gets a step of size ~lr
    instead of lr*wd - a 10x larger pull towards zero here."""
    p = Parameter(np.array([1.0]))
    opt = Adam([p], lr=0.1, weight_decay=0.1)
    p.grad = np.zeros(1)
    opt.step()
    coupled_step = 1.0 - p.data.item()
    adamw_step = 0.1 * 0.1
    assert coupled_step == pytest.approx(0.1, rel=1e-3)
    assert coupled_step / adamw_step == pytest.approx(10.0, rel=1e-2)


def test_effective_decay_under_coupled_adam_depends_on_gradient_scale() -> None:
    """The core pathology: identical `weight_decay`, different gradient scale,
    different amount of shrinkage. Under AdamW the shrinkage is invariant."""
    shrink_coupled, shrink_decoupled = [], []
    for grad_scale in (1e-2, 1e2):
        pc, pd = Parameter(np.array([1.0])), Parameter(np.array([1.0]))
        oc = Adam([pc], lr=0.01, weight_decay=0.5)
        od = AdamW([pd], lr=0.01, weight_decay=0.5)
        for _ in range(50):
            pc.grad = np.array([grad_scale])
            pd.grad = np.array([grad_scale])
            oc.step()
            od.step()
        # Isolate decay by subtracting a no-decay run with the same gradients.
        pc0, pd0 = Parameter(np.array([1.0])), Parameter(np.array([1.0]))
        oc0, od0 = Adam([pc0], lr=0.01), AdamW([pd0], lr=0.01, weight_decay=0.0)
        for _ in range(50):
            pc0.grad = np.array([grad_scale])
            pd0.grad = np.array([grad_scale])
            oc0.step()
            od0.step()
        shrink_coupled.append((pc0.data - pc.data).item())
        shrink_decoupled.append((pd0.data - pd.data).item())

    ratio_coupled = shrink_coupled[0] / shrink_coupled[1]
    ratio_decoupled = shrink_decoupled[0] / shrink_decoupled[1]
    assert ratio_decoupled == pytest.approx(1.0, abs=1e-9), "AdamW decay must be scale-free"
    assert abs(ratio_coupled - 1.0) > 0.05, "coupled decay should vary with gradient scale"


def test_adamw_is_a_subclass_but_flips_the_decoupled_flag() -> None:
    p = Parameter(np.zeros(1))
    assert AdamW([p], lr=0.1).decoupled is True
    assert Adam([p], lr=0.1).decoupled is False
