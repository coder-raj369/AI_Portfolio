"""RMSNorm, written out rather than imported from `nn.RMSNorm`."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    r""":math:`y = \gamma \cdot x / \sqrt{\mathrm{mean}(x^2) + \epsilon}`.

    The `.float()` cast is not cosmetic. Under bf16 autocast the mean of squares
    over a few thousand channels loses enough precision to shift the normalisation
    measurably, so every production implementation computes the statistic in fp32
    and casts back - the same discipline measured in `00-foundations/nn-primitives`.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * normed).to(dtype)
