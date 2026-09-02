from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.view(1, 1, t, t), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_hidden: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_hidden)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
    """A tiny decoder-only model used to prove the full pretrain/SFT/DPO pipeline."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 32,
        d_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model=d_model, n_heads=n_heads, d_hidden=d_hidden) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.dim() != 2:
            raise ValueError("TinyDecoderLM expects a 2D tensor of token ids")
        b, t = input_ids.shape
        if t > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len :]
            t = self.max_seq_len

        positions = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt_ids: list[int] | Tensor, max_new_tokens: int = 8) -> list[int]:
        if isinstance(prompt_ids, list):
            prompt = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        else:
            prompt = prompt_ids.to(dtype=torch.long)

        for _ in range(max_new_tokens):
            if prompt.shape[1] > self.max_seq_len:
                break
            logits = self(prompt[:, -self.max_seq_len :])
            next_token = logits[:, -1].argmax(dim=-1)
            prompt = torch.cat([prompt, next_token.unsqueeze(1)], dim=1)
            if int(next_token.item()) == 0:
                break
        return prompt[0].tolist()

    def sequence_log_probs(self, tokens: Tensor) -> Tensor:
        if tokens.dim() != 1:
            raise ValueError("sequence_log_probs expects a 1D token sequence")
        if len(tokens) <= 1:
            return torch.zeros((), device=tokens.device)
        logits = self(tokens[:-1].unsqueeze(0))[:, :, :]
        log_probs = F.log_softmax(logits, dim=-1)
        target = tokens[1:].unsqueeze(0)
        gathered = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        return gathered.sum(dim=-1)
