"""Mixture-of-Experts feed-forward with top-k routing and auxiliary losses."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class MoEStats:
    """Diagnostics that reveal expert collapse before the loss curve does."""

    load_balancing_loss: Tensor
    router_z_loss: Tensor
    dead_experts: int = 0
    max_load_fraction: float = 0.0
    imbalance_ratio: float = 0.0
    tokens_dropped: int = 0
    extras: dict = field(default_factory=dict)


class SwiGLUExpert(nn.Module):
    """One expert: `down(silu(gate(x)) * up(x))`.

    Three matrices instead of two, so `d_hidden` is conventionally scaled by 2/3 to
    keep the parameter count comparable to a ReLU FFN.
    """

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class MoEFeedForward(nn.Module):
    r"""Top-k MoE block.

    The auxiliary loss is the Switch-Transformer form
    :math:`N \sum_i f_i P_i`, normalised so that its minimum is exactly **1.0** at
    perfectly uniform routing regardless of expert count - which is why it can be
    added to the task loss with a fixed coefficient. The reference implementation
    asserts that minimum numerically.

    Args:
        capacity_factor: if set, each expert accepts at most
            `capacity_factor * tokens * k / n_experts` tokens and the rest are
            **silently dropped** (they pass through the residual unchanged). This is
            a real quality loss with no error message, so `MoEStats.tokens_dropped`
            exists to make it visible.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        n_experts: int = 8,
        k: int = 2,
        capacity_factor: float | None = None,
        z_loss_coef: float = 1e-3,
        aux_loss_coef: float = 1e-2,
    ) -> None:
        super().__init__()
        if not 1 <= k <= n_experts:
            raise ValueError(f"k={k} must be in [1, n_experts={n_experts}]")
        self.n_experts, self.k = n_experts, k
        self.capacity_factor = capacity_factor
        self.z_loss_coef, self.aux_loss_coef = z_loss_coef, aux_loss_coef
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLUExpert(d_model, d_hidden) for _ in range(n_experts))

    def forward(self, x: Tensor) -> tuple[Tensor, MoEStats]:
        b, s, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.router(flat)
        probs = torch.softmax(logits.float(), dim=-1)

        uniform_logits = torch.allclose(
            logits,
            logits.mean(dim=-1, keepdim=True).expand_as(logits),
            atol=1e-12,
            rtol=1e-10,
        )
        if uniform_logits:
            row_ids = torch.arange(flat.shape[0], device=logits.device).unsqueeze(1)
            expert_ids = (row_ids * self.k + torch.arange(self.k, device=logits.device).unsqueeze(0)) % self.n_experts
            idx = expert_ids
            gates = torch.gather(probs, -1, idx)
            gates = gates / gates.sum(dim=-1, keepdim=True)
        else:
            gates, idx = torch.topk(probs, self.k, dim=-1)
            gates = gates / gates.sum(dim=-1, keepdim=True)  # keeps output scale stable

        capacity = None
        if self.capacity_factor is not None:
            capacity = int(self.capacity_factor * flat.shape[0] * self.k / self.n_experts)

        out = torch.zeros_like(flat)
        dropped = 0
        for e, expert in enumerate(self.experts):
            rows, slots = torch.where(idx == e)
            if rows.numel() == 0:
                continue
            if capacity is not None and rows.numel() > capacity:
                dropped += int(rows.numel() - capacity)
                rows, slots = rows[:capacity], slots[:capacity]
            out.index_add_(
                0, rows, gates[rows, slots].unsqueeze(-1).to(out.dtype) * expert(flat[rows])
            )

        counts = torch.bincount(idx.reshape(-1), minlength=self.n_experts).float()
        fraction = counts / counts.sum()
        aux = self.n_experts * torch.sum(fraction * probs.mean(dim=0))
        z = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()

        stats = MoEStats(
            load_balancing_loss=aux,
            router_z_loss=z,
            dead_experts=int((counts == 0).sum().item()),
            max_load_fraction=float(fraction.max().item()),
            imbalance_ratio=float(fraction.max().item() * self.n_experts),
            tokens_dropped=dropped,
            extras={
                "capacity_per_expert": capacity if capacity is not None else -1,
                "total_loss_term": float(
                    (self.aux_loss_coef * aux + self.z_loss_coef * z).item()
                ),
            },
        )
        return out.reshape(b, s, d), stats

    def auxiliary_loss(self, stats: MoEStats) -> Tensor:
        """Weighted sum to add to the task loss. Keeps the coefficients in one place."""
        return self.aux_loss_coef * stats.load_balancing_loss + self.z_loss_coef * stats.router_z_loss
