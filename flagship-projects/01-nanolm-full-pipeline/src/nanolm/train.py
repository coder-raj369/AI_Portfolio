from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TrainingResult:
    final_loss: float
    mean_loss: float
    checkpoints: list[float]


def _prepare_batch(sequences: list[list[int]], pad_id: int = 0, max_seq_len: int | None = None) -> tuple[Tensor, Tensor]:
    clipped = [seq[:max_seq_len] if max_seq_len is not None else seq for seq in sequences]
    max_len = max(len(seq) for seq in clipped)
    x = torch.full((len(clipped), max_len), pad_id, dtype=torch.long)
    y = torch.full((len(clipped), max_len), -100, dtype=torch.long)
    for i, seq in enumerate(clipped):
        length = len(seq)
        x[i, :length] = torch.tensor(seq, dtype=torch.long)
        if length > 1:
            y[i, : length - 1] = torch.tensor(seq[1:], dtype=torch.long)
    return x, y


def _next_token_loss(model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    logits = model(x)
    logits = logits[:, :-1, :].reshape(-1, model.vocab_size)
    targets = y[:, 1:].reshape(-1)
    return F.cross_entropy(logits, targets, ignore_index=-100)


def train_base_model(
    model: nn.Module,
    sequences: list[list[int]],
    *,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 3e-3,
    pad_id: int = 0,
) -> TrainingResult:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    max_seq_len = getattr(model, "max_seq_len", None)

    for _ in range(epochs):
        random.shuffle(sequences)
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            x, y = _prepare_batch(batch, pad_id=pad_id, max_seq_len=max_seq_len)
            optimizer.zero_grad()
            loss = _next_token_loss(model, x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

    return TrainingResult(
        final_loss=float(losses[-1]),
        mean_loss=float(sum(losses) / len(losses)),
        checkpoints=losses,
    )


def sft_finetune(
    model: nn.Module,
    prompt_answer_pairs: list[tuple[list[int], list[int]]],
    *,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 2e-3,
    pad_id: int = 0,
) -> TrainingResult:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    max_seq_len = getattr(model, "max_seq_len", None)

    for _ in range(epochs):
        random.shuffle(prompt_answer_pairs)
        for start in range(0, len(prompt_answer_pairs), batch_size):
            batch = prompt_answer_pairs[start : start + batch_size]
            x_list: list[list[int]] = []
            label_spans: list[tuple[int, list[int]]] = []
            for prompt, answer in batch:
                if max_seq_len is not None and len(prompt) + len(answer) > max_seq_len:
                    answer = answer[:max_seq_len]
                    prompt = prompt[-max(0, max_seq_len - len(answer)) :]
                seq = prompt + answer
                x_list.append(seq)
                label_spans.append((len(prompt), answer))
            x, _ = _prepare_batch(x_list, pad_id=pad_id, max_seq_len=max_seq_len)
            labels = torch.full_like(x, -100)
            for i, (prompt_len, answer_seq) in enumerate(label_spans):
                if answer_seq:
                    labels[i, prompt_len : prompt_len + len(answer_seq)] = torch.tensor(answer_seq, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(x)
            logits = logits.reshape(-1, model.vocab_size)
            targets = labels.reshape(-1)
            loss = F.cross_entropy(logits, targets, ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

    return TrainingResult(
        final_loss=float(losses[-1]),
        mean_loss=float(sum(losses) / len(losses)),
        checkpoints=losses,
    )


def evaluate_model(model: nn.Module, prompt_answer_pairs: list[tuple[list[int], list[int]]], tokenizer=None) -> dict[str, float]:
    correct = 0
    total = 0
    max_seq_len = getattr(model, "max_seq_len", None)
    for prompt, answer in prompt_answer_pairs:
        input_ids = prompt[-max_seq_len:] if max_seq_len is not None else prompt
        with torch.no_grad():
            logits = model(torch.tensor(input_ids, dtype=torch.long).unsqueeze(0))
        next_token = logits[:, -1].argmax(dim=-1).item()
        if next_token in answer[:1]:
            correct += 1
        total += 1
    if total == 0:
        return {"accuracy": 0.0}
    return {"accuracy": correct / total}


def dpo_loss(model: nn.Module, ref_model: nn.Module, chosen: Tensor, rejected: Tensor, beta: float = 0.1) -> Tensor:
    chosen_logp = model.sequence_log_probs(chosen)
    rejected_logp = model.sequence_log_probs(rejected)
    ref_chosen_logp = ref_model.sequence_log_probs(chosen)
    ref_rejected_logp = ref_model.sequence_log_probs(rejected)
    delta = (chosen_logp - ref_chosen_logp) - (rejected_logp - ref_rejected_logp)
    return -F.logsigmoid(beta * delta)


def train_dpo(model: nn.Module, preference_pairs: list[tuple[list[int], list[int], list[int]]], *, epochs: int = 20, lr: float = 1e-3) -> TrainingResult:
    ref_model = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []

    for _ in range(epochs):
        random.shuffle(preference_pairs)
        for prompt, chosen, rejected in preference_pairs:
            chosen_tensor = torch.tensor(chosen, dtype=torch.long)
            rejected_tensor = torch.tensor(rejected, dtype=torch.long)
            optimizer.zero_grad()
            loss = dpo_loss(model, ref_model, chosen_tensor, rejected_tensor)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

    return TrainingResult(
        final_loss=float(losses[-1]),
        mean_loss=float(sum(losses) / len(losses)),
        checkpoints=losses,
    )


def _sample_completion(model: nn.Module, prompt: list[int], max_new_tokens: int, temperature: float) -> list[int]:
    """Sample one completion while keeping the model inside its context window."""
    device = next(model.parameters()).device
    tokens = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    completion: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(tokens[:, -model.max_seq_len :])[:, -1]
        probabilities = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        completion.append(int(next_token.item()))
        tokens = torch.cat([tokens, next_token], dim=1)
    return completion


def _completion_log_prob(model: nn.Module, prompt: list[int], completion: list[int]) -> Tensor:
    """Return the summed log probability of completion tokens conditioned on prompt."""
    device = next(model.parameters()).device
    sequence = torch.tensor(prompt + completion, dtype=torch.long, device=device)
    context = sequence[:-1]
    if len(context) > model.max_seq_len:
        context = context[-model.max_seq_len :]
    logits = model(context.unsqueeze(0))[0, -len(completion) :]
    targets = sequence[-len(completion) :]
    return F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).sum()


def grpo_loss(rewards: Tensor, completion_log_probs: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute a group-relative policy-gradient loss from one prompt's samples."""
    if rewards.ndim != 1 or completion_log_probs.ndim != 1:
        raise ValueError("GRPO inputs must be one-dimensional group tensors")
    if rewards.shape != completion_log_probs.shape:
        raise ValueError("rewards and completion_log_probs must have the same shape")
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)
    return -(advantages.detach() * completion_log_probs).mean()


def train_grpo(
    model: nn.Module,
    prompts: list[list[int]],
    reward_fn: Callable[[list[int], list[int]], float],
    *,
    epochs: int = 10,
    group_size: int = 4,
    max_new_tokens: int = 4,
    temperature: float = 1.0,
    lr: float = 1e-3,
) -> TrainingResult:
    """Train with verifier rewards and group-relative policy optimization."""
    if group_size < 2:
        raise ValueError("group_size must be at least 2 for relative advantages")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(epochs):
        random.shuffle(prompts)
        for prompt in prompts:
            completions = [
                _sample_completion(model, prompt, max_new_tokens, temperature)
                for _ in range(group_size)
            ]
            rewards = torch.tensor(
                [reward_fn(prompt, completion) for completion in completions],
                dtype=torch.float32,
                device=next(model.parameters()).device,
            )
            log_probs = torch.stack([
                _completion_log_prob(model, prompt, completion) for completion in completions
            ])
            optimizer.zero_grad()
            loss = grpo_loss(rewards, log_probs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

    return TrainingResult(
        final_loss=float(losses[-1]),
        mean_loss=float(sum(losses) / len(losses)),
        checkpoints=losses,
    )
