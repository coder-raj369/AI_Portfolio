"""GQA and the KV cache, checked by equivalence rather than by inspection."""

from __future__ import annotations

import numpy as np
import pytest

from nanoformer.reference import GroupedQueryAttention, KVCache, alibi_bias, repeat_kv

RNG = np.random.default_rng(1)


def test_kv_cache_incremental_decoding_matches_a_full_forward() -> None:
    """The whole point of a KV cache: token-by-token decoding must produce bitwise
    (to float tolerance) the same outputs as one full causal forward pass."""
    attn = GroupedQueryAttention(64, n_heads=8, n_kv_heads=2, max_seq=64, seed=0)
    x = RNG.standard_normal((1, 12, 64))
    full = attn.forward(x, causal=True)

    cache = KVCache(1, attn.n_kv_heads, attn.head_dim, 64)
    steps = [attn.forward(x[:, t : t + 1], cache=cache, causal=True) for t in range(12)]
    np.testing.assert_allclose(np.concatenate(steps, axis=1), full, atol=1e-12)


def test_prefill_then_decode_matches_a_full_forward() -> None:
    """The realistic serving path: a batched prefill followed by single-token steps."""
    attn = GroupedQueryAttention(64, n_heads=8, n_kv_heads=4, max_seq=64, seed=0)
    x = RNG.standard_normal((2, 12, 64))
    full = attn.forward(x, causal=True)

    cache = KVCache(2, attn.n_kv_heads, attn.head_dim, 64)
    prefill = attn.forward(x[:, :8], cache=cache, causal=True)
    decode = [attn.forward(x[:, t : t + 1], cache=cache, causal=True) for t in range(8, 12)]
    got = np.concatenate([prefill, *decode], axis=1)
    np.testing.assert_allclose(got, full, atol=1e-12)


def test_cache_overflow_raises_instead_of_corrupting() -> None:
    attn = GroupedQueryAttention(32, n_heads=4, n_kv_heads=1, max_seq=32, seed=0)
    cache = KVCache(1, 1, attn.head_dim, max_seq=4)
    with pytest.raises(ValueError, match="overflow"):
        attn.forward(RNG.standard_normal((1, 5, 32)), cache=cache)


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4, 8])
def test_gqa_shapes_and_kv_cost_scale_with_n_kv_heads(n_kv_heads: int) -> None:
    attn = GroupedQueryAttention(64, n_heads=8, n_kv_heads=n_kv_heads, seed=0)
    out = attn.forward(RNG.standard_normal((2, 6, 64)))
    assert out.shape == (2, 6, 64)
    assert attn.n_rep == 8 // n_kv_heads
    # 2 (k and v) * n_kv_heads * head_dim * 2 bytes
    assert attn.kv_cache_bytes_per_token_fp16() == 2 * n_kv_heads * 8 * 2


def test_multi_query_attention_uses_one_eighth_the_cache_of_mha() -> None:
    mha = GroupedQueryAttention(64, n_heads=8, n_kv_heads=8, seed=0)
    mqa = GroupedQueryAttention(64, n_heads=8, n_kv_heads=1, seed=0)
    ratio = mha.kv_cache_bytes_per_token_fp16() / mqa.kv_cache_bytes_per_token_fp16()
    assert ratio == pytest.approx(8.0)


def test_repeat_kv_duplicates_each_head_contiguously() -> None:
    x = np.arange(2 * 2 * 3 * 4, dtype=float).reshape(2, 2, 3, 4)
    out = repeat_kv(x, 3)
    assert out.shape == (2, 6, 3, 4)
    # heads 0,1,2 all come from source head 0
    for i in range(3):
        np.testing.assert_array_equal(out[:, i], x[:, 0])
    for i in range(3, 6):
        np.testing.assert_array_equal(out[:, i], x[:, 1])


def test_repeat_kv_with_n_rep_one_is_the_identity() -> None:
    x = RNG.standard_normal((1, 4, 5, 8))
    np.testing.assert_array_equal(repeat_kv(x, 1), x)


def test_causal_attention_ignores_the_future() -> None:
    attn = GroupedQueryAttention(32, n_heads=4, n_kv_heads=2, seed=0)
    x = RNG.standard_normal((1, 6, 32))
    base = attn.forward(x, causal=True)
    perturbed = x.copy()
    perturbed[0, -1] += 50.0
    after = attn.forward(perturbed, causal=True)
    np.testing.assert_allclose(base[0, :-1], after[0, :-1], atol=1e-12)
    assert not np.allclose(base[0, -1], after[0, -1])


def test_alibi_bias_biases_attention_towards_recent_tokens() -> None:
    """With ALiBi and no RoPE, the average attended position must move later."""
    attn = GroupedQueryAttention(32, n_heads=4, n_kv_heads=4, seed=0)
    x = RNG.standard_normal((1, 16, 32))
    plain = attn.forward(x, causal=True, use_rope=False)
    biased = attn.forward(
        x, causal=True, use_rope=False, alibi=alibi_bias(4, 16, 16)
    )
    assert not np.allclose(plain, biased)


def test_invalid_head_configurations_raise() -> None:
    with pytest.raises(ValueError, match="n_heads"):
        GroupedQueryAttention(64, n_heads=6, seed=0)
    with pytest.raises(ValueError, match="n_kv_heads"):
        GroupedQueryAttention(64, n_heads=8, n_kv_heads=3, seed=0)
