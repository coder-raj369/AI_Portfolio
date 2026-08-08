"""Transformer, MoE and hybrid SSM-attention blocks, plus a small decoder LM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nanoformer.torch_impl.attention import GroupedQueryAttention, KVCache
from nanoformer.torch_impl.moe import MoEFeedForward, MoEStats
from nanoformer.torch_impl.norm import RMSNorm
from nanoformer.torch_impl.ssm import SelectiveSSM


class SwiGLUFeedForward(nn.Module):
    """Dense FFN with SwiGLU. `d_hidden` defaults to `8/3 * d_model`, rounded.

    The 8/3 (rather than 4) keeps the parameter count of a three-matrix SwiGLU FFN
    roughly equal to a two-matrix 4x ReLU FFN - the convention Llama introduced.
    """

    def __init__(self, d_model: int, d_hidden: int | None = None) -> None:
        super().__init__()
        d_hidden = d_hidden or int(round(8 * d_model / 3 / 8) * 8)
        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


@dataclass
class ModelConfig:
    """Configuration for `TinyDecoderLM`.

    Attributes:
        layer_pattern: per-layer mixer, `"attn"` or `"ssm"`. A hybrid stack
            interleaves them - e.g. `["ssm", "ssm", "attn"] * n` - because SSM layers
            are O(seq) and cheap but cannot do precise long-range token lookup, while
            attention layers can. Production hybrids keep a small minority of
            attention layers for exactly that reason.
        moe_layers: indices that use an MoE FFN instead of a dense one. Common
            practice is every other layer, not all of them: MoE multiplies parameter
            count and memory traffic, and the marginal gain per MoE layer falls off.
    """

    vocab_size: int = 512
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    d_state: int = 16
    max_seq: int = 256
    layer_pattern: tuple[str, ...] = ("attn", "attn", "attn", "attn")
    moe_layers: tuple[int, ...] = ()
    n_experts: int = 4
    moe_k: int = 2

    def __post_init__(self) -> None:
        if len(self.layer_pattern) != self.n_layers:
            raise ValueError(
                f"layer_pattern has {len(self.layer_pattern)} entries "
                f"but n_layers={self.n_layers}"
            )
        bad = set(self.layer_pattern) - {"attn", "ssm"}
        if bad:
            raise ValueError(f"unknown mixer(s) {sorted(bad)}; use 'attn' or 'ssm'")
        if any(i >= self.n_layers for i in self.moe_layers):
            raise ValueError(f"moe_layers {self.moe_layers} out of range for {self.n_layers}")


class Block(nn.Module):
    """Pre-norm residual block: `x + mixer(norm(x))`, then `x + ffn(norm(x))`.

    Pre-norm (not post-norm) because it is what trains without a warmup-heavy
    schedule at depth: the residual stream stays an identity path, so gradients reach
    early layers directly.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.kind = cfg.layer_pattern[layer_idx]
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        if self.kind == "attn":
            self.mixer = GroupedQueryAttention(
                cfg.d_model, cfg.n_heads, cfg.n_kv_heads, max_seq=cfg.max_seq
            )
        else:
            self.mixer = SelectiveSSM(cfg.d_model, d_state=cfg.d_state)
        self.is_moe = layer_idx in cfg.moe_layers
        if self.is_moe:
            self.ffn = MoEFeedForward(
                cfg.d_model, int(round(8 * cfg.d_model / 3 / 8) * 8),
                n_experts=cfg.n_experts, k=cfg.moe_k,
            )
        else:
            self.ffn = SwiGLUFeedForward(cfg.d_model)

    def forward(
        self, x: Tensor, cache: KVCache | None = None
    ) -> tuple[Tensor, MoEStats | None]:
        normed = self.norm1(x)
        if self.kind == "attn":
            x = x + self.mixer(normed, cache=cache, causal=True)
        else:
            x = x + self.mixer(normed)  # the recurrence is causal by construction
        if self.is_moe:
            ffn_out, stats = self.ffn(self.norm2(x))
            return x + ffn_out, stats
        return x + self.ffn(self.norm2(x)), None


class TinyDecoderLM(nn.Module):
    """A small decoder-only LM: RoPE + RMSNorm + GQA + SwiGLU, optional MoE/SSM layers.

    Weight tying between the embedding and the output head is on by default: it saves
    `vocab_size * d_model` parameters, which at small scale is most of the model.
    """

    def __init__(self, cfg: ModelConfig, tie_weights: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.embed.weight
        self.apply(self._init)
        # GPT-2 residual rescaling: scale output projections by 1/sqrt(2 * n_layers)
        # so residual-stream variance does not grow with depth.
        scale = (2 * cfg.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith(("w_o.weight", "out_proj.weight", "down.weight")):
                with torch.no_grad():
                    p.mul_(scale)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self, tokens: Tensor, caches: list[KVCache] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Args: `tokens` of shape `(batch, seq)`.

        Returns:
            `(logits, aux_loss)` where `aux_loss` is the summed MoE auxiliary terms
            (zero if no MoE layers). Returning it explicitly - rather than stashing it
            on the module - is what stops it being silently dropped from the backward
            pass, which is the most common way an MoE quietly collapses.
        """
        x = self.embed(tokens)
        aux = x.new_zeros(())
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None and block.kind == "attn" else None
            x, stats = block(x, cache=cache)
            if stats is not None:
                aux = aux + block.ffn.auxiliary_loss(stats)
        return self.head(self.norm_f(x)), aux

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def active_parameters_per_token(self) -> int:
        """Parameters actually used per token - differs from the total under MoE."""
        total = self.num_parameters()
        for block in self.blocks:
            if block.is_moe:
                experts = block.ffn.experts
                per_expert = sum(p.numel() for p in experts[0].parameters())
                total -= (len(experts) - block.ffn.k) * per_expert
        return total

    @torch.no_grad()
    def generate(
        self, prompt: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> Tensor:
        """Greedy / sampled decoding with a KV cache for the attention layers."""
        self.eval()
        b = prompt.shape[0]
        caches = [
            KVCache(b, blk.mixer.n_kv_heads, blk.mixer.head_dim, self.cfg.max_seq,
                    device=prompt.device, dtype=self.embed.weight.dtype)
            if blk.kind == "attn" else None
            for blk in self.blocks
        ]
        # An SSM layer here carries no decode state, so a hybrid stack must re-run
        # the whole prefix each step. Stateful SSM decoding (carrying `h` forward) is
        # a real optimisation and is listed as a limitation in the README rather than
        # pretended away.
        has_ssm = any(b.kind == "ssm" for b in self.blocks)
        tokens = prompt
        logits, _ = self.forward(prompt, caches=caches)
        for _ in range(max_new_tokens):
            last = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                kth = torch.topk(last, top_k, dim=-1).values[:, -1:]
                last = last.masked_fill(last < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(last, dim=-1), 1)
            tokens = torch.cat([tokens, nxt], dim=1)
            if has_ssm:
                for c in caches:
                    if c is not None:
                        c.reset()
                logits, _ = self.forward(tokens, caches=caches)
            else:
                logits, _ = self.forward(nxt, caches=caches)
        return tokens
