"""Attention: equivalence to a naive loop, causality, and why the scale matters."""

from __future__ import annotations

import numpy as np
import pytest

from nnprim.attention import (
    causal_mask,
    merge_heads,
    multi_head_attention,
    naive_multi_head_attention,
    scaled_dot_product_attention,
    split_heads,
)

RNG = np.random.default_rng(9)


def _weights(d_model: int, n: int = 4) -> list[np.ndarray]:
    return [RNG.standard_normal((d_model, d_model)) * (1.0 / np.sqrt(d_model)) for _ in range(n)]


@pytest.mark.parametrize("causal", [False, True])
def test_batched_implementation_matches_the_naive_loop(causal: bool) -> None:
    x = RNG.standard_normal((3, 7, 16))
    w = _weights(16)
    fast, _ = multi_head_attention(x, *w, n_heads=4, causal=causal)
    slow = naive_multi_head_attention(x, *w, n_heads=4, causal=causal)
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-12)


def test_split_and_merge_heads_roundtrip() -> None:
    x = RNG.standard_normal((2, 5, 12))
    np.testing.assert_array_equal(merge_heads(split_heads(x, 3)), x)


def test_split_heads_rejects_indivisible_d_model() -> None:
    with pytest.raises(ValueError, match="divisible"):
        split_heads(RNG.standard_normal((1, 4, 10)), n_heads=3)


def test_attention_weights_are_a_distribution_over_keys() -> None:
    q, k, v = (RNG.standard_normal((2, 4, 6, 8)) for _ in range(3))
    _, w = scaled_dot_product_attention(q, k, v)
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-12)
    assert (w >= 0).all()


def test_causal_attention_ignores_the_future() -> None:
    """Perturb the last token and assert every earlier output is bit-identical.

    This is the test that actually proves causality; checking the mask's shape
    does not, and an off-by-one in the triangle leaks exactly one future token.
    """
    x = RNG.standard_normal((1, 6, 16))
    w = _weights(16)
    base, _ = multi_head_attention(x, *w, n_heads=4, causal=True)
    perturbed = x.copy()
    perturbed[0, -1] += 100.0
    after, _ = multi_head_attention(perturbed, *w, n_heads=4, causal=True)
    np.testing.assert_array_equal(base[0, :-1], after[0, :-1])
    assert not np.allclose(base[0, -1], after[0, -1])


def test_non_causal_attention_does_see_the_future() -> None:
    x = RNG.standard_normal((1, 6, 16))
    w = _weights(16)
    base, _ = multi_head_attention(x, *w, n_heads=4, causal=False)
    perturbed = x.copy()
    perturbed[0, -1] += 100.0
    after, _ = multi_head_attention(perturbed, *w, n_heads=4, causal=False)
    assert not np.allclose(base[0, 0], after[0, 0])


def test_causal_mask_is_strictly_lower_triangular_inclusive() -> None:
    m = causal_mask(4)
    assert m[0].tolist() == [True, False, False, False]
    assert m[3].tolist() == [True, True, True, True]


def test_attention_is_permutation_equivariant_without_positions() -> None:
    """Shuffling the tokens shuffles the outputs identically - i.e. attention has
    no notion of order on its own. This is the reason positional encodings exist,
    and it is checkable rather than merely quotable."""
    x = RNG.standard_normal((1, 6, 16))
    w = _weights(16)
    perm = np.array([3, 0, 5, 1, 4, 2])
    out_a, _ = multi_head_attention(x, *w, n_heads=4, causal=False)
    out_b, _ = multi_head_attention(x[:, perm], *w, n_heads=4, causal=False)
    np.testing.assert_allclose(out_a[0][perm], out_b[0], rtol=1e-11, atol=1e-11)


def test_missing_the_sqrt_dk_scale_saturates_the_softmax() -> None:
    """Quantifies why the 1/sqrt(d_k) factor is load-bearing: without it the
    attention distribution collapses towards one-hot and its gradient vanishes."""
    d_head = 64
    q = RNG.standard_normal((1, 1, 32, d_head))
    k = RNG.standard_normal((1, 1, 32, d_head))
    v = RNG.standard_normal((1, 1, 32, d_head))
    _, scaled = scaled_dot_product_attention(q, k, v)
    unscaled_scores = q @ np.swapaxes(k, -1, -2)  # deliberately unscaled
    from nnprim.functional import softmax

    unscaled = softmax(unscaled_scores, axis=-1)

    def entropy(w: np.ndarray) -> float:
        return float(-(w * np.log(w + 1e-12)).sum(axis=-1).mean())

    assert entropy(scaled) > 2.5            # broad: 2.98 nats vs ln(32) = 3.47 for uniform
    assert entropy(unscaled) < 0.7          # collapsed to 0.54 nats, i.e. near one-hot
    assert unscaled.max() > 0.95


def test_attention_output_is_a_convex_combination_of_values() -> None:
    v = RNG.standard_normal((1, 1, 5, 4))
    q = RNG.standard_normal((1, 1, 5, 4))
    out, _ = scaled_dot_product_attention(q, q, v)
    assert (out <= v.max(axis=2, keepdims=True) + 1e-12).all()
    assert (out >= v.min(axis=2, keepdims=True) - 1e-12).all()
