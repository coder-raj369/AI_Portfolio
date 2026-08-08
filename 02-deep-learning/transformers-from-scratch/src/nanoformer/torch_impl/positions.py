"""RoPE and ALiBi in PyTorch, mirroring `nanoformer.reference.positions`.

Kept structurally identical to the NumPy reference so the equivalence tests are
comparing arithmetic, not two different algorithms that happen to agree.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE tables, registered as non-persistent buffers.

    Non-persistent because they are a pure function of `(head_dim, max_seq, theta)`:
    serialising them bloats checkpoints and, worse, silently pins the context length
    of any model loaded from one.
    """

    def __init__(self, head_dim: int, max_seq: int = 4096, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(half, dtype=torch.float64) / half))
        angles = torch.outer(torch.arange(max_seq, dtype=torch.float64), inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Rotate `x` of shape `(batch, heads, seq, head_dim)` by its positions.

        Args:
            offset: absolute position of `x[..., 0, :]`. Non-zero during cached
                decoding; passing 0 there is the classic RoPE bug and shows up only
                as degraded long-context behaviour, never as a crash.
        """
        seq, half = x.shape[-2], x.shape[-1] // 2
        if offset + seq > self.cos.shape[0]:
            raise ValueError(
                f"position {offset + seq} exceeds precomputed max_seq={self.cos.shape[0]}"
            )
        cos = self.cos[offset : offset + seq, :half].to(dtype=x.dtype, device=x.device)
        sin = self.sin[offset : offset + seq, :half].to(dtype=x.dtype, device=x.device)
        x1, x2 = x[..., :half], x[..., half:]
        # Interleaved-halves convention (channel i pairs with i + d/2), as used by
        # Llama-family checkpoints. GPT-NeoX pairs adjacent channels; the two differ
        # by a permutation of the head dim, so mixing them corrupts a loaded model.
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def alibi_slopes(n_heads: int, device=None, dtype=torch.float32) -> Tensor:
    """Geometric slope ladder from the ALiBi paper (handles non-powers of two)."""
    def ladder(n: int) -> Tensor:
        start = 2.0 ** (-(2.0 ** -(torch.log2(torch.tensor(float(n))) - 3).item()))
        return torch.tensor(
            [start ** (k + 1) for k in range(n)], device=device, dtype=torch.float64
        )

    if n_heads & (n_heads - 1) == 0:
        slopes = ladder(n_heads)
    else:
        closest = 2 ** int(torch.floor(torch.log2(torch.tensor(float(n_heads)))).item())
        slopes = torch.cat([ladder(closest), ladder(2 * closest)[0::2][: n_heads - closest]])
    return slopes.to(dtype=dtype)


def alibi_bias(
    n_heads: int, seq_q: int, seq_k: int, offset: int = 0, device=None, dtype=torch.float32
) -> Tensor:
    """Additive attention bias `-slope_h * |m - n|`, shaped `(n_heads, seq_q, seq_k)`."""
    q_pos = torch.arange(offset, offset + seq_q, device=device, dtype=dtype).unsqueeze(1)
    k_pos = torch.arange(seq_k, device=device, dtype=dtype).unsqueeze(0)
    distance = (q_pos - k_pos).abs()
    return -alibi_slopes(n_heads, device=device, dtype=dtype).view(-1, 1, 1) * distance
