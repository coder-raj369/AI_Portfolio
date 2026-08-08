"""MoE routing: the auxiliary loss is calibrated, and collapse is detectable."""

from __future__ import annotations

import numpy as np
import pytest

from nanoformer.reference import (
    MoEFeedForward,
    expert_utilisation,
    load_balancing_loss,
    router_z_loss,
    top_k_router,
)

RNG = np.random.default_rng(2)


def test_load_balancing_loss_is_exactly_one_at_uniform_routing() -> None:
    """The normalisation by N is what makes this loss usable with a fixed
    coefficient: its minimum is 1.0 for any number of experts."""
    for n_experts in (4, 8, 16, 64):
        probs = np.full((512, n_experts), 1.0 / n_experts)
        idx = (np.arange(512 * 2) % n_experts).reshape(512, 2)
        assert load_balancing_loss(probs, idx) == pytest.approx(1.0, rel=1e-12)


def test_load_balancing_loss_equals_n_under_total_collapse() -> None:
    """All tokens to one expert, router fully confident: the loss saturates at N,
    so the gradient signal is largest exactly when collapse is worst."""
    for n_experts in (4, 8, 16):
        probs = np.eye(n_experts)[np.zeros(300, dtype=int)]
        idx = np.zeros((300, 2), dtype=int)
        assert load_balancing_loss(probs, idx) == pytest.approx(float(n_experts), rel=1e-12)


def test_load_balancing_loss_is_monotone_in_imbalance() -> None:
    n_experts, tokens = 8, 800
    losses = []
    for skew in (0.0, 0.5, 0.9):
        # `skew` of the tokens go to expert 0, the rest spread uniformly.
        n_skewed = int(skew * tokens)
        idx = np.concatenate([
            np.zeros((n_skewed, 1), dtype=int),
            (np.arange(tokens - n_skewed) % n_experts).reshape(-1, 1),
        ])
        probs = np.zeros((tokens, n_experts))
        probs[np.arange(tokens), idx[:, 0]] = 1.0
        losses.append(load_balancing_loss(probs, idx))
    assert losses == sorted(losses), losses


def test_top_k_router_selects_the_highest_probability_experts() -> None:
    logits = np.array([[0.1, 5.0, 0.2, 3.0]])
    idx, gates, probs = top_k_router(logits, k=2)
    assert idx[0].tolist() == [1, 3]
    assert gates.sum() == pytest.approx(1.0)
    assert probs.sum() == pytest.approx(1.0)


def test_gate_renormalisation_keeps_the_output_scale_stable() -> None:
    """Without renormalisation, a hesitant router shrinks the block's output, which
    silently changes the residual-stream scale during training."""
    logits = RNG.standard_normal((64, 8)) * 0.01  # near-uniform: gates ~ 1/8 each
    _, renormed, _ = top_k_router(logits, k=2, renormalise=True)
    _, raw, _ = top_k_router(logits, k=2, renormalise=False)
    np.testing.assert_allclose(renormed.sum(axis=-1), 1.0, rtol=1e-12)
    assert raw.sum(axis=-1).mean() < 0.4


def test_router_k_is_validated() -> None:
    with pytest.raises(ValueError, match="k="):
        top_k_router(RNG.standard_normal((4, 8)), k=9)


def test_router_z_loss_penalises_large_logits() -> None:
    small = router_z_loss(RNG.standard_normal((100, 8)))
    large = router_z_loss(RNG.standard_normal((100, 8)) * 50.0)
    assert large > small * 100


def test_expert_utilisation_reports_dead_experts_and_entropy() -> None:
    idx = np.zeros((100, 2), dtype=int)  # everything to expert 0
    stats = expert_utilisation(idx, n_experts=8)
    assert stats["dead_experts"] == 7
    assert stats["max_load_fraction"] == pytest.approx(1.0)
    assert stats["load_entropy_nats"] == pytest.approx(0.0)
    assert stats["imbalance_ratio"] == pytest.approx(8.0)

    balanced = (np.arange(800) % 8).reshape(400, 2)
    stats = expert_utilisation(balanced, n_experts=8)
    assert stats["dead_experts"] == 0
    assert stats["load_entropy_nats"] == pytest.approx(stats["uniform_entropy_nats"], rel=1e-9)
    assert stats["imbalance_ratio"] == pytest.approx(1.0)


def test_moe_block_output_shape_and_active_parameter_accounting() -> None:
    moe = MoEFeedForward(d_model=32, d_hidden=64, n_experts=8, k=2, seed=0)
    out, stats = moe.forward(RNG.standard_normal((4, 16, 32)))
    assert out.shape == (4, 16, 32)
    # The core MoE claim: total capacity grows with n_experts, per-token cost with k.
    assert stats["total_expert_params"] == 8 * 3 * 32 * 64
    assert stats["active_expert_params_per_token"] == 2 * 3 * 32 * 64
    ratio = stats["total_expert_params"] / stats["active_expert_params_per_token"]
    assert ratio == pytest.approx(4.0)


def test_untrained_router_is_already_roughly_balanced() -> None:
    """At init the router is near-random, so load is close to uniform. Collapse is a
    *training* dynamic - which is why the auxiliary loss must be present from step 0,
    not added after the loss curve looks wrong."""
    moe = MoEFeedForward(32, 64, n_experts=8, k=2, seed=0)
    _, stats = moe.forward(RNG.standard_normal((8, 64, 32)))
    assert stats["dead_experts"] == 0
    assert stats["load_balancing_loss"] < 1.2
    assert stats["imbalance_ratio"] < 1.6


def test_capacity_limiting_drops_tokens_and_reports_it() -> None:
    """Capacity dropping is a silent quality loss, so it must be observable."""
    moe = MoEFeedForward(32, 64, n_experts=8, k=2, capacity_factor=0.25, seed=0)
    _, stats = moe.forward(RNG.standard_normal((4, 64, 32)))
    assert stats["tokens_dropped"] > 0
    assert stats["capacity_per_expert"] > 0

    unlimited = MoEFeedForward(32, 64, n_experts=8, k=2, seed=0)
    _, stats2 = unlimited.forward(RNG.standard_normal((4, 64, 32)))
    assert stats2["tokens_dropped"] == 0


def test_k_equals_one_is_switch_routing() -> None:
    moe = MoEFeedForward(32, 64, n_experts=4, k=1, seed=0)
    out, stats = moe.forward(RNG.standard_normal((2, 8, 32)))
    assert out.shape == (2, 8, 32)
    assert stats["active_expert_params_per_token"] == stats["total_expert_params"] // 4
