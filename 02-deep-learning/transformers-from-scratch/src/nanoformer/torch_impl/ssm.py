"""Mamba-style selective state-space block in PyTorch.

The selective scan is written as an explicit Python loop over the sequence. That is
deliberately the slow version: it is obviously correct, it is what the NumPy
reference is checked against, and it makes the point that the *algorithm* is a
recurrence while the *speed* comes entirely from a fused kernel. A chunked
formulation (`selective_scan_chunked`) is provided alongside to show the structure
a real associative-scan kernel exploits.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def selective_scan_sequential(
    x: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor | None = None
) -> Tensor:
    r"""`h_t = exp(delta_t A) h_{t-1} + delta_t B_t x_t`, `y_t = C_t h_t + D x_t`.

    Args:
        x: `(b, l, d_inner)`.
        delta: `(b, l, d_inner)`, positive.
        A: `(d_inner, d_state)`, negative (log-domain state matrix).
        B, C: `(b, l, d_state)` - input-dependent, which is what "selective" means
            and what forbids the FFT convolution trick that S4 relies on.
        D: `(d_inner,)` skip, or None.
    """
    b, l, d_inner = x.shape
    h = x.new_zeros(b, d_inner, A.shape[1])
    outputs = []
    for t in range(l):
        dA = torch.exp(delta[:, t].unsqueeze(-1) * A.unsqueeze(0))
        dBx = delta[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
        h = dA * h + dBx
        outputs.append(torch.einsum("bin,bn->bi", h, C[:, t]))
    y = torch.stack(outputs, dim=1)
    return y + D.view(1, 1, -1) * x if D is not None else y


def selective_scan_chunked(
    x: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
    chunk: int = 16,
) -> Tensor:
    """Same result via cumulative products within chunks.

    Trades memory (an `(L, L)` weight matrix per chunk) for parallelism across the
    chunk, which is the shape of the computation a GPU scan kernel performs. Done in
    the log domain so the cumulative products cannot underflow.
    """
    b, l, d_inner = x.shape
    h = x.new_zeros(b, d_inner, A.shape[1])
    out = torch.zeros_like(x)
    for start in range(0, l, chunk):
        stop = min(start + chunk, l)
        ln = stop - start
        d = delta[:, start:stop].unsqueeze(-1)
        log_dA = d * A.view(1, 1, *A.shape)
        cum = torch.cumsum(log_dA, dim=1)
        dBx = d * B[:, start:stop].unsqueeze(2) * x[:, start:stop].unsqueeze(-1)

        carried = torch.exp(cum) * h.unsqueeze(1)
        diff = cum.unsqueeze(2) - cum.unsqueeze(1)
        mask = torch.tril(torch.ones(ln, ln, dtype=torch.bool, device=x.device))
        weights = torch.where(
            mask.view(1, ln, ln, 1, 1), torch.exp(diff.clamp(max=0.0)), torch.zeros_like(diff)
        )
        h_all = carried + torch.einsum("btsin,bsin->btin", weights, dBx)
        out[:, start:stop] = torch.einsum("btin,btn->bti", h_all, C[:, start:stop])
        h = h_all[:, -1]
    return out + D.view(1, 1, -1) * x if D is not None else out


class SelectiveSSM(nn.Module):
    """Mamba-style block: gated selective scan with input-dependent dt, B and C.

    Omits the depthwise causal convolution of the real Mamba block: it adds local
    mixing but nothing to the selectivity mechanism this module exists to
    demonstrate, and leaving it out keeps the scan-equivalence test unambiguous.
    """

    def __init__(
        self, d_model: int, d_state: int = 16, expand: int = 2, dt_init: float = -2.0
    ) -> None:
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.constant_(self.dt_proj.bias, dt_init)  # softplus(-2) ~ 0.13
        # S4D-real initialisation: A = -(1..d_state), stable and well-conditioned.
        # Stored as a plain parameter in log-free form; A must stay negative for the
        # recurrence to contract, which `A_log`-style parameterisations enforce
        # structurally. Here it is initialised negative and left trainable.
        self.A = nn.Parameter(
            -torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, u: Tensor, chunked: bool = False) -> Tensor:
        """Args: `u` of shape `(batch, seq, d_model)`."""
        x, gate = self.in_proj(u).chunk(2, dim=-1)
        delta = torch.nn.functional.softplus(self.dt_proj(x))
        B, C = self.B_proj(x), self.C_proj(x)
        scan = selective_scan_chunked if chunked else selective_scan_sequential
        y = scan(x, delta, self.A, B, C, self.D)
        return self.out_proj(y * torch.nn.functional.silu(gate))
