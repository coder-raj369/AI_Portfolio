r"""RoPE and ALiBi as executable NumPy reference implementations.

Both encode position *inside* the attention logit rather than by adding a vector
to the token embedding, and both are defined by a property that can be checked
numerically rather than trusted:

- **RoPE**: the logit :math:`q_m^\top k_n` depends only on :math:`m - n`.
- **ALiBi**: the bias added to the logit is exactly :math:`-\text{slope}\,|m - n|`.

These references are the ground truth the PyTorch implementations in
`nanoformer.torch_impl` are tested against.
"""

from __future__ import annotations

import numpy as np


def rope_frequencies(head_dim: int, seq_len: int, theta: float = 10_000.0) -> tuple:
    r"""Precompute the cos/sin tables for rotary position embedding.

    Args:
        head_dim: per-head dimension. Must be even - RoPE rotates *pairs* of
            channels, so an odd dimension has an unpairable leftover.
        seq_len: number of positions to precompute.
        theta: the base of the geometric frequency ladder. 10,000 is the original
            value; long-context models raise it (or rescale positions) so the
            lowest frequency still completes less than one full turn over the
            context - the mechanism behind "RoPE scaling".

    Returns:
        `(cos, sin)`, each shaped `(seq_len, head_dim // 2)`.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    angles = np.outer(np.arange(seq_len, dtype=np.float64), inv_freq)
    return np.cos(angles), np.sin(angles)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, offset: int = 0) -> np.ndarray:
    r"""Rotate `x` by its position angle.

    Args:
        x: `(..., seq, head_dim)`.
        cos, sin: tables from `rope_frequencies`.
        offset: position of the first element of `x`. Non-zero during cached
            decoding, where `x` is a single token at absolute position `offset` -
            getting this wrong is the single most common RoPE bug, and it only
            shows up as slightly worse long-context behaviour.

    Returns:
        The rotated array, same shape as `x`.

    Uses the *interleaved-halves* convention (channel `i` pairs with `i + d/2`),
    which is what Llama-family checkpoints use. GPT-NeoX pairs adjacent channels
    instead; the two are related by a fixed permutation of the head dimension, so
    mixing them silently degrades a loaded checkpoint.
    """
    seq = x.shape[-2]
    half = x.shape[-1] // 2
    c = cos[offset : offset + seq, :half]
    s = sin[offset : offset + seq, :half]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def alibi_slopes(n_heads: int) -> np.ndarray:
    r"""Geometric slope ladder from the ALiBi paper.

    For a power-of-two head count the slopes are :math:`2^{-8k/n}` for
    :math:`k = 1..n`. Non-powers of two are handled by taking the ladder for the
    next lower power of two and interleaving a finer ladder - reproducing the
    reference implementation exactly rather than inventing an interpolation.
    """
    def ladder(n: int) -> np.ndarray:
        start = 2.0 ** (-(2.0 ** -(np.log2(n) - 3)))
        return start ** np.arange(1, n + 1, dtype=np.float64)

    if n_heads & (n_heads - 1) == 0:
        return ladder(n_heads)
    closest = 2 ** int(np.floor(np.log2(n_heads)))
    base = ladder(closest)
    extra = ladder(2 * closest)[0::2][: n_heads - closest]
    return np.concatenate([base, extra])


def alibi_bias(n_heads: int, seq_q: int, seq_k: int, offset: int = 0) -> np.ndarray:
    r"""Additive attention bias :math:`-\text{slope}_h |m - n|`.

    Returns:
        `(n_heads, seq_q, seq_k)`.

    Unlike RoPE this needs no learned parameters and no table sized to the
    training context, which is why ALiBi extrapolates past its training length -
    at the cost of a fixed recency prior baked into every head.
    """
    q_pos = np.arange(offset, offset + seq_q, dtype=np.float64)[:, None]
    k_pos = np.arange(seq_k, dtype=np.float64)[None, :]
    distance = np.abs(q_pos - k_pos)
    return -alibi_slopes(n_heads)[:, None, None] * distance[None, :, :]
