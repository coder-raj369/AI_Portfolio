"""Grouped-query attention with a KV cache, in NumPy.

Two things are worth implementing from scratch here rather than importing:

1. **GQA.** `n_kv_heads < n_heads` shrinks the KV cache by `n_heads / n_kv_heads`,
   which is the dominant memory cost during long-context decoding. Setting
   `n_kv_heads == n_heads` must recover exact MHA, and `n_kv_heads == 1` gives
   multi-query attention - so one implementation covers all three, and the tests
   assert the degenerate cases agree with the general one.
2. **The KV cache.** Its whole justification is that incremental decoding produces
   *identical* logits to a full forward pass over the same prefix. That is an
   equivalence property, so it gets an equivalence test.
"""

from __future__ import annotations

import numpy as np

from nanoformer.reference.positions import apply_rope, rope_frequencies

NEG_INF = -1e9


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """Expand `(batch, n_kv_heads, seq, d)` to `(batch, n_kv_heads * n_rep, seq, d)`.

    Each KV head is repeated for the query heads that share it. This materialises
    the expansion for clarity; a production kernel indexes into the cache instead,
    which is where the memory saving actually lands.
    """
    if n_rep == 1:
        return x
    batch, n_kv, seq, d = x.shape
    return np.repeat(x, n_rep, axis=1).reshape(batch, n_kv * n_rep, seq, d)


class KVCache:
    """Append-only key/value cache for autoregressive decoding."""

    def __init__(self, batch: int, n_kv_heads: int, head_dim: int, max_seq: int) -> None:
        self.k = np.zeros((batch, n_kv_heads, max_seq, head_dim))
        self.v = np.zeros((batch, n_kv_heads, max_seq, head_dim))
        self.length = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append new keys/values and return the full cached history."""
        new = k.shape[2]
        if self.length + new > self.k.shape[2]:
            raise ValueError(
                f"cache overflow: {self.length} + {new} > {self.k.shape[2]}; "
                "grow max_seq or evict"
            )
        self.k[:, :, self.length : self.length + new] = k
        self.v[:, :, self.length : self.length + new] = v
        self.length += new
        return self.k[:, :, : self.length], self.v[:, :, : self.length]

    @property
    def bytes_fp16(self) -> int:
        """Memory footprint at fp16 - the number that decides max batch size."""
        return 2 * self.k[:, :, : self.length].size * 2


class GroupedQueryAttention:
    """Multi-head / grouped-query / multi-query attention with RoPE and a KV cache.

    Attributes:
        n_heads: number of query heads.
        n_kv_heads: number of key/value heads; must divide `n_heads`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        max_seq: int = 512,
        rope_theta: float = 10_000.0,
        seed: int = 0,
    ) -> None:
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        if n_heads % n_kv_heads:
            raise ValueError(f"n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        kv_dim = n_kv_heads * self.head_dim
        self.w_q = rng.normal(0, scale, (d_model, d_model))
        self.w_k = rng.normal(0, scale, (d_model, kv_dim))
        self.w_v = rng.normal(0, scale, (d_model, kv_dim))
        self.w_o = rng.normal(0, scale, (d_model, d_model))
        self.cos, self.sin = rope_frequencies(self.head_dim, max_seq, rope_theta)

    def _split(self, x: np.ndarray, n_heads: int) -> np.ndarray:
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(
        self,
        x: np.ndarray,
        cache: KVCache | None = None,
        causal: bool = True,
        alibi: np.ndarray | None = None,
        use_rope: bool = True,
    ) -> np.ndarray:
        """Run attention over `x` of shape `(batch, seq, d_model)`.

        Args:
            cache: if given, `x` is treated as a continuation and the cache
                supplies the history. The query positions are offset by
                `cache.length`, which is what keeps RoPE consistent between the
                prefill and decode paths.
            causal: apply a causal mask. With a cache, only the new queries are
                masked against the (already causal) history.
            alibi: optional additive bias, `(n_heads, seq_q, seq_k)`.
            use_rope: disable to isolate the effect of positions in tests.
        """
        batch, seq, _ = x.shape
        offset = cache.length if cache is not None else 0

        q = self._split(x @ self.w_q, self.n_heads)
        k = self._split(x @ self.w_k, self.n_kv_heads)
        v = self._split(x @ self.w_v, self.n_kv_heads)

        if use_rope:
            q = apply_rope(q, self.cos, self.sin, offset=offset)
            k = apply_rope(k, self.cos, self.sin, offset=offset)

        if cache is not None:
            k, v = cache.append(k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(self.head_dim)
        if alibi is not None:
            scores = scores + alibi[None, :, :, : scores.shape[-1]]
        if causal:
            seq_k = scores.shape[-1]
            q_idx = np.arange(offset, offset + seq)[:, None]
            k_idx = np.arange(seq_k)[None, :]
            scores = np.where(q_idx >= k_idx, scores, NEG_INF)

        ctx = softmax(scores, axis=-1) @ v
        merged = ctx.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        return merged @ self.w_o

    def kv_cache_bytes_per_token_fp16(self) -> int:
        """Per-token KV cache cost: `2 * n_kv_heads * head_dim * 2` bytes."""
        return 2 * self.n_kv_heads * self.head_dim * 2
