r"""A Mamba-style selective state-space model, in NumPy.

The core recurrence (Mamba's S6) is

.. math::
    h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \qquad y_t = C_t h_t + D x_t

with :math:`\bar{A}_t = \exp(\Delta_t A)` and :math:`\bar{B}_t = \Delta_t B_t`.
What makes it *selective* - and what makes it stronger than a classical LTI SSM -
is that :math:`\Delta_t`, :math:`B_t` and :math:`C_t` are functions of the input,
so the model can decide per token how much to remember.

That input dependence is exactly what breaks the convolution trick used by S4, so
this module implements two paths and asserts they agree:

- `selective_scan_sequential`: the literal recurrence, O(seq) steps, obviously
  correct and the ground truth.
- `selective_scan_chunked`: the same result computed over chunks using cumulative
  products, which is the structure a parallel/associative scan kernel exploits.

Also implemented is `lti_scan_via_convolution`, which shows the *non*-selective
case reduces to an FIR convolution - the property that lets S4 run in
:math:`O(L \log L)` and which selectivity gives up.
"""

from __future__ import annotations

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    """`log(1 + exp(x))`, computed without overflowing for large `x`."""
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def selective_scan_sequential(
    x: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray | None = None,
) -> np.ndarray:
    r"""Ground-truth selective scan, written as the literal recurrence.

    Args:
        x: `(batch, seq, d_inner)` input.
        delta: `(batch, seq, d_inner)` positive step sizes.
        A: `(d_inner, d_state)` log-domain state matrix (diagonal, negative).
        B: `(batch, seq, d_state)` input-dependent input projection.
        C: `(batch, seq, d_state)` input-dependent output projection.
        D: `(d_inner,)` skip connection, or None.

    Returns:
        `(batch, seq, d_inner)`.
    """
    batch, seq, d_inner = x.shape
    d_state = A.shape[1]
    h = np.zeros((batch, d_inner, d_state))
    out = np.zeros_like(x)
    for t in range(seq):
        # dA: (batch, d_inner, d_state); exp of a negative number stays in (0, 1),
        # which is what keeps the recurrence stable without any explicit gating.
        dA = np.exp(delta[:, t, :, None] * A[None, :, :])
        dBx = delta[:, t, :, None] * B[:, t, None, :] * x[:, t, :, None]
        h = dA * h + dBx
        out[:, t, :] = np.einsum("bin,bn->bi", h, C[:, t, :])
    if D is not None:
        out = out + D[None, None, :] * x
    return out


def selective_scan_chunked(
    x: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray | None = None,
    chunk: int = 16,
) -> np.ndarray:
    r"""Same result, computed chunk-wise with cumulative products.

    Within a chunk the recurrence is unrolled as

    .. math::
        h_t = \left(\prod_{s\le t}\bar{A}_s\right) h_{-1}
            + \sum_{s \le t} \frac{\prod_{u \le t}\bar A_u}{\prod_{u \le s}\bar A_u}
              \bar{B}_s x_s

    which turns the sequential dependency into a cumulative product plus a matmul -
    the structure a GPU associative-scan kernel actually implements. Done in the
    log domain so the cumulative products cannot underflow to zero.
    """
    batch, seq, d_inner = x.shape
    d_state = A.shape[1]
    h = np.zeros((batch, d_inner, d_state))
    out = np.zeros_like(x)

    for start in range(0, seq, chunk):
        stop = min(start + chunk, seq)
        d = delta[:, start:stop, :, None]              # (b, L, i, 1)
        log_dA = d * A[None, None, :, :]               # (b, L, i, n), <= 0
        cum = np.cumsum(log_dA, axis=1)                # inclusive cumulative log-product
        dBx = d * B[:, start:stop, None, :] * x[:, start:stop, :, None]

        # Contribution of the carried-in state, and of each earlier step in the chunk.
        carried = np.exp(cum) * h[:, None, :, :]
        # weights[t, s] = exp(cum[t] - cum[s]) for s <= t, else 0
        diff = cum[:, :, None, :, :] - cum[:, None, :, :, :]
        mask = np.tril(np.ones((stop - start, stop - start), dtype=bool))[
            None, :, :, None, None
        ]
        weights = np.where(mask, np.exp(np.minimum(diff, 0.0)), 0.0)
        local = np.einsum("btsin,bsin->btin", weights, dBx)

        h_all = carried + local                        # (b, L, i, n)
        out[:, start:stop, :] = np.einsum("btin,btn->bti", h_all, C[:, start:stop, :])
        h = h_all[:, -1]

    if D is not None:
        out = out + D[None, None, :] * x
    return out


def lti_scan_via_convolution(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    r"""The *non*-selective case: constant A, B, C reduce to an FIR convolution.

    With :math:`h_t = a h_{t-1} + b x_t` and :math:`y_t = c h_t`, unrolling gives
    :math:`y_t = \sum_{s \le t} c\,b\,a^{t-s} x_s`, i.e. a convolution with kernel
    :math:`c b a^{k}`. This is why S4 can run in :math:`O(L \log L)` via FFT, and
    why Mamba's input-dependent parameters give that up in exchange for
    selectivity.

    Args:
        x: `(batch, seq)`; a, b, c: scalars as 0-d arrays.
    """
    seq = x.shape[1]
    kernel = float(c) * float(b) * float(a) ** np.arange(seq)
    out = np.zeros_like(x)
    for t in range(seq):
        out[:, t] = x[:, : t + 1] @ kernel[: t + 1][::-1]
    return out


class SelectiveSSMBlock:
    """A Mamba-style block: input projection -> selective scan -> output projection.

    Deliberately omits the depthwise causal convolution from the real Mamba block.
    It contributes local mixing but nothing to the selectivity story this module
    exists to demonstrate, and leaving it out keeps the scan equivalence test
    unambiguous about what it is testing.
    """

    def __init__(
        self, d_model: int, d_state: int = 16, expand: int = 2, seed: int = 0
    ) -> None:
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.w_in = rng.normal(0, scale, (d_model, 2 * self.d_inner))  # x and gate
        self.w_dt = rng.normal(0, scale, (self.d_inner, self.d_inner))
        self.w_B = rng.normal(0, scale, (self.d_inner, d_state))
        self.w_C = rng.normal(0, scale, (self.d_inner, d_state))
        self.dt_bias = np.full((self.d_inner,), -2.0)  # softplus(-2) ~ 0.13
        # S4D-real initialisation: A = -(1..d_state), stable and well-conditioned.
        self.A = -np.tile(np.arange(1, d_state + 1, dtype=np.float64), (self.d_inner, 1))
        self.D = np.ones(self.d_inner)
        self.w_out = rng.normal(0, 1.0 / np.sqrt(self.d_inner), (self.d_inner, d_model))

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        pos = x >= 0
        z = np.exp(-np.abs(x))
        return x * np.where(pos, 1.0 / (1.0 + z), z / (1.0 + z))

    def forward(self, u: np.ndarray, chunked: bool = False) -> np.ndarray:
        """Args: `u` of shape `(batch, seq, d_model)`."""
        proj = u @ self.w_in
        x, gate = proj[..., : self.d_inner], proj[..., self.d_inner :]
        delta = softplus(x @ self.w_dt + self.dt_bias)
        B = x @ self.w_B
        C = x @ self.w_C
        scan = selective_scan_chunked if chunked else selective_scan_sequential
        y = scan(x, delta, self.A, B, C, self.D)
        return (y * self._silu(gate)) @ self.w_out
