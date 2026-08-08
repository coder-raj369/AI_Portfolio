"""Grouped-query attention with a KV cache, from raw tensor ops.

No `nn.MultiheadAttention` and no `F.scaled_dot_product_attention` in the forward
path - though `benchmark.py` measures against the latter, because knowing how much
a fused kernel buys you is part of understanding the layer.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from nanoformer.torch_impl.positions import RotaryEmbedding


class KVCache:
    """Pre-allocated append-only key/value cache.

    Pre-allocating `max_seq` and slicing beats growing by `torch.cat` on every
    token: concatenation reallocates and copies the whole cache each step, which
    turns decoding into O(seq^2) memory traffic.
    """

    def __init__(
        self,
        batch: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq: int,
        device=None,
        dtype=torch.float32,
    ) -> None:
        shape = (batch, n_kv_heads, max_seq, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
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

    def reset(self) -> None:
        self.length = 0

    def nbytes(self) -> int:
        used = self.k[:, :, : self.length]
        return 2 * used.numel() * used.element_size()


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """`(b, n_kv, s, d)` -> `(b, n_kv * n_rep, s, d)`, repeating each KV head."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x.unsqueeze(2).expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


class GroupedQueryAttention(nn.Module):
    """MHA / GQA / MQA in one module, selected by `n_kv_heads`.

    `n_kv_heads == n_heads` is exact MHA; `n_kv_heads == 1` is multi-query. The KV
    cache shrinks by `n_heads / n_kv_heads`, which during long-context decoding is
    the difference between fitting a batch and not.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        max_seq: int = 4096,
        rope_theta: float = 10_000.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        if n_heads % n_kv_heads:
            raise ValueError(f"n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}")
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        kv_dim = n_kv_heads * self.head_dim

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, kv_dim, bias=bias)
        self.w_v = nn.Linear(d_model, kv_dim, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.rope = RotaryEmbedding(self.head_dim, max_seq=max_seq, theta=rope_theta)

    def forward(
        self,
        x: Tensor,
        cache: KVCache | None = None,
        causal: bool = True,
        alibi: Tensor | None = None,
        use_rope: bool = True,
    ) -> Tensor:
        """Args: `x` of shape `(batch, seq, d_model)`."""
        b, s, _ = x.shape
        offset = cache.length if cache is not None else 0

        q = self.w_q(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if use_rope:
            q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.append(k, v)

        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if alibi is not None:
            scores = scores + alibi[None, :, :, : scores.shape[-1]]
        if causal:
            seq_k = scores.shape[-1]
            q_idx = torch.arange(offset, offset + s, device=x.device).unsqueeze(1)
            k_idx = torch.arange(seq_k, device=x.device).unsqueeze(0)
            # -1e9 rather than -inf: a fully masked row of -inf gives nan after
            # softmax, and masked rows do occur with padding.
            scores = scores.masked_fill(q_idx < k_idx, -1e9)

        ctx = torch.softmax(scores, dim=-1) @ v
        return self.w_o(ctx.transpose(1, 2).reshape(b, s, self.d_model))

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """`2 * n_kv_heads * head_dim * dtype_bytes` - the long-context bottleneck."""
        return 2 * self.n_kv_heads * self.head_dim * dtype_bytes
