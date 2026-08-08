r"""Mixture-of-Experts routing and the load-balancing loss, in NumPy.

The routing arithmetic is where MoE actually goes wrong. Two failure modes matter
and both are measurable:

1. **Expert collapse.** Without an auxiliary loss the router funnels almost all
   tokens to a few experts; the rest never receive gradient and stay dead.
2. **Capacity dropping.** With a finite per-expert capacity, tokens past the limit
   are silently dropped and pass through unchanged. Quality loss with no error.

The auxiliary loss implemented here is the Switch-Transformer form

.. math::
    L_{aux} = N \sum_{i=1}^{N} f_i \cdot P_i

where :math:`f_i` is the fraction of tokens routed to expert *i* and :math:`P_i`
the mean router probability for it. The normalisation by :math:`N` makes the
*minimum* exactly 1.0 at a perfectly uniform distribution, independent of the
number of experts - which is why it can be added to the task loss with a fixed
coefficient. That minimum is asserted in the tests.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def top_k_router(
    logits: np.ndarray, k: int = 2, renormalise: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route each token to its top-`k` experts.

    Args:
        logits: `(tokens, n_experts)` router logits.
        k: experts per token.
        renormalise: rescale the selected gate values to sum to 1. Without this the
            block's output magnitude depends on how confident the router happened
            to be, which shifts the residual-stream scale during training.

    Returns:
        `(expert_indices, gates, probs)` where the first two are `(tokens, k)` and
        `probs` is the full `(tokens, n_experts)` distribution.
    """
    if not 1 <= k <= logits.shape[-1]:
        raise ValueError(f"k={k} must be in [1, n_experts={logits.shape[-1]}]")
    probs = softmax(logits, axis=-1)
    idx = np.argsort(-probs, axis=-1, kind="stable")[:, :k]
    gates = np.take_along_axis(probs, idx, axis=-1)
    if renormalise:
        gates = gates / gates.sum(axis=-1, keepdims=True)
    return idx, gates, probs


def load_balancing_loss(probs: np.ndarray, expert_indices: np.ndarray) -> float:
    r"""Switch-Transformer auxiliary loss, normalised so uniform routing gives 1.0."""
    n_experts = probs.shape[-1]
    counts = np.bincount(expert_indices.reshape(-1), minlength=n_experts)
    fraction = counts / counts.sum()
    mean_prob = probs.mean(axis=0)
    return float(n_experts * np.sum(fraction * mean_prob))


def router_z_loss(logits: np.ndarray) -> float:
    r"""Router z-loss: :math:`\text{mean}(\log \sum_i e^{z_i})^2`.

    Penalises large router logits. Introduced in ST-MoE because routers trained in
    bf16 drift to huge logits, and the resulting softmax is numerically unstable
    in exactly the way `nn-primitives` measures.
    """
    m = logits.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))).squeeze(-1)
    return float(np.mean(lse**2))


def expert_utilisation(expert_indices: np.ndarray, n_experts: int) -> dict[str, float]:
    """Diagnostics that reveal collapse before the loss curve does."""
    counts = np.bincount(expert_indices.reshape(-1), minlength=n_experts)
    fraction = counts / max(counts.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = float(-np.sum(np.where(fraction > 0, fraction * np.log(fraction), 0.0)))
    return {
        "n_experts": n_experts,
        "dead_experts": int((counts == 0).sum()),
        "max_load_fraction": float(fraction.max()),
        "min_load_fraction": float(fraction.min()),
        "load_entropy_nats": entropy,
        "uniform_entropy_nats": float(np.log(n_experts)),
        # Ratio of worst-loaded expert to the average: the number that decides how
        # badly a distributed MoE step is straggler-bound.
        "imbalance_ratio": float(fraction.max() * n_experts),
    }


class MoEFeedForward:
    """Top-k MoE block with SwiGLU experts and optional capacity limiting."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        n_experts: int = 8,
        k: int = 2,
        capacity_factor: float | None = None,
        seed: int = 0,
    ) -> None:
        self.d_model, self.n_experts, self.k = d_model, n_experts, k
        self.capacity_factor = capacity_factor
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.w_router = rng.normal(0, scale, (d_model, n_experts))
        # SwiGLU experts: gate, up, down. Three matrices instead of two, so d_hidden
        # is conventionally scaled by 2/3 to keep the parameter count comparable.
        self.w_gate = rng.normal(0, scale, (n_experts, d_model, d_hidden))
        self.w_up = rng.normal(0, scale, (n_experts, d_model, d_hidden))
        self.w_down = rng.normal(0, 1.0 / np.sqrt(d_hidden), (n_experts, d_hidden, d_model))

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        pos = x >= 0
        z = np.exp(-np.abs(x))
        sig = np.where(pos, 1.0 / (1.0 + z), z / (1.0 + z))
        return x * sig

    def _expert(self, x: np.ndarray, e: int) -> np.ndarray:
        return (self._silu(x @ self.w_gate[e]) * (x @ self.w_up[e])) @ self.w_down[e]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Args: `x` of shape `(batch, seq, d_model)`. Returns `(out, stats)`."""
        batch, seq, d = x.shape
        flat = x.reshape(-1, d)
        idx, gates, probs = top_k_router(flat @ self.w_router, self.k)

        capacity = None
        dropped = 0
        if self.capacity_factor is not None:
            capacity = int(self.capacity_factor * flat.shape[0] * self.k / self.n_experts)

        out = np.zeros_like(flat)
        for e in range(self.n_experts):
            rows, slots = np.nonzero(idx == e)
            if rows.size == 0:
                continue
            if capacity is not None and rows.size > capacity:
                # First-come-first-served, matching the reference implementations.
                dropped += rows.size - capacity
                rows, slots = rows[:capacity], slots[:capacity]
            out[rows] += gates[rows, slots][:, None] * self._expert(flat[rows], e)

        stats = expert_utilisation(idx, self.n_experts)
        stats["load_balancing_loss"] = load_balancing_loss(probs, idx)
        stats["router_z_loss"] = router_z_loss(flat @ self.w_router)
        stats["tokens_dropped"] = int(dropped)
        stats["capacity_per_expert"] = capacity if capacity is not None else -1
        # Active parameters per token vs total: the entire point of MoE.
        per_expert = 3 * self.d_model * self.w_gate.shape[2]
        stats["total_expert_params"] = self.n_experts * per_expert
        stats["active_expert_params_per_token"] = self.k * per_expert
        return out.reshape(batch, seq, d), stats
