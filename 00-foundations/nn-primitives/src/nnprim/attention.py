"""Multi-head attention from raw tensor operations.

No `nn.MultiheadAttention`, no fused kernel: the reshape/transpose choreography,
the 1/sqrt(d_head) scale and the causal mask are all written out, because that
choreography is exactly what gets asked about and exactly what the library hides.
"""

from __future__ import annotations

import numpy as np

from nnprim.functional import softmax

NEG_INF = -1e9  # not -np.inf: keeps fully-masked rows from producing nan in softmax


def causal_mask(seq_len: int) -> np.ndarray:
    """Boolean `(seq, seq)` mask, `True` where attention is **allowed**."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r""":math:`\mathrm{softmax}(QK^\top/\sqrt{d_k})V`.

    Args:
        q: `(..., seq_q, d_head)`.
        k, v: `(..., seq_kv, d_head)`.
        mask: broadcastable boolean mask, `True` = keep.
        causal: apply a lower-triangular mask (requires `seq_q == seq_kv`).

    Returns:
        `(output, attention_weights)`.

    The :math:`1/\sqrt{d_k}` scale is not cosmetic: for unit-variance q and k the
    dot product has variance `d_k`, so at `d_k=64` the logits reach ~8 and the
    softmax saturates into a near one-hot distribution whose gradient is ~0.
    """
    d_head = q.shape[-1]
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_head)
    if causal:
        cm = causal_mask(scores.shape[-1])
        scores = np.where(cm, scores, NEG_INF)
    if mask is not None:
        scores = np.where(mask, scores, NEG_INF)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    """`(batch, seq, d_model)` -> `(batch, heads, seq, d_head)`.

    The transpose is the step people skip. Without it the head dimension stays
    interleaved with the sequence dimension and every head attends to a stripe
    of the wrong positions - which still trains, just worse, so it is a silent bug.
    """
    batch, seq, d_model = x.shape
    if d_model % n_heads:
        raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
    d_head = d_model // n_heads
    return x.reshape(batch, seq, n_heads, d_head).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    """`(batch, heads, seq, d_head)` -> `(batch, seq, d_model)`."""
    batch, heads, seq, d_head = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, seq, heads * d_head)


def multi_head_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    n_heads: int,
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched multi-head self-attention.

    Args:
        x: `(batch, seq, d_model)`.
        w_q, w_k, w_v, w_o: `(d_model, d_model)` projections.

    Returns:
        `(output, attention_weights)` with weights shaped `(batch, heads, seq, seq)`.
    """
    q = split_heads(x @ w_q, n_heads)
    k = split_heads(x @ w_k, n_heads)
    v = split_heads(x @ w_v, n_heads)
    ctx, weights = scaled_dot_product_attention(q, k, v, causal=causal)
    return merge_heads(ctx) @ w_o, weights


def naive_multi_head_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    n_heads: int,
    causal: bool = False,
) -> np.ndarray:
    """Reference implementation: an explicit Python loop over batch and heads.

    Obviously slow, and that is the point - it is the ground truth the batched
    reshape/transpose version is tested against, so a wrong transpose cannot
    hide behind plausible-looking output.
    """
    batch, seq, d_model = x.shape
    d_head = d_model // n_heads
    out = np.zeros((batch, seq, d_model))
    for b in range(batch):
        heads = []
        for h in range(n_heads):
            sl = slice(h * d_head, (h + 1) * d_head)
            q = x[b] @ w_q[:, sl]
            k = x[b] @ w_k[:, sl]
            v = x[b] @ w_v[:, sl]
            scores = (q @ k.T) / np.sqrt(d_head)
            if causal:
                scores = np.where(causal_mask(seq), scores, NEG_INF)
            heads.append(softmax(scores, axis=-1) @ v)
        out[b] = np.concatenate(heads, axis=-1) @ w_o
    return out
