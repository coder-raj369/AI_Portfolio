from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from nanolm.data import (
    build_instruction_pairs,
    build_pretrain_corpus,
    build_preference_pairs,
    train_tokenizer,
)
from nanolm.model import TinyDecoderLM
from nanolm.train import evaluate_model, sft_finetune, train_base_model, train_dpo, train_grpo

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def train_demo() -> None:
    random.seed(0)
    torch.manual_seed(0)
    corpus = build_pretrain_corpus()
    tokenizer = train_tokenizer(corpus, vocab_size=512)
    pretrain_seqs = [list(tokenizer.encode(text, allowed_special="all")) for text in corpus.split("\n") if text.strip()]
    model = TinyDecoderLM(vocab_size=tokenizer.vocab_size, d_model=32, n_layers=2, n_heads=4, max_seq_len=32, d_hidden=64)
    base = train_base_model(model, pretrain_seqs, epochs=8, batch_size=8, lr=5e-3)

    ins = []
    for prompt, answer in build_instruction_pairs():
        ins.append((tokenizer.encode(prompt, allowed_special="all"), tokenizer.encode(answer, allowed_special="all")))
    sft = sft_finetune(model, ins, epochs=15, batch_size=4, lr=2e-3)

    pref = []
    for prompt, chosen, rejected in build_preference_pairs():
        pref.append((tokenizer.encode(prompt, allowed_special="all"), tokenizer.encode(chosen, allowed_special="all"), tokenizer.encode(rejected, allowed_special="all")))
    dpo = train_dpo(model, pref, epochs=10, lr=1e-3)

    prompts = [tokenizer.encode(prompt, allowed_special="all") for prompt, _ in build_instruction_pairs()]

    def reward_fn(prompt_ids: list[int], completion_ids: list[int]) -> float:
        prompt_text = tokenizer.decode(prompt_ids)
        expected = next(answer for prompt, answer in build_instruction_pairs() if prompt == prompt_text)
        completion_text = tokenizer.decode(completion_ids).strip()
        if completion_text.startswith(expected):
            return 1.0
        overlap = sum(character in completion_text for character in set(expected.lower()))
        return 0.05 * overlap / max(len(set(expected.lower())), 1)

    grpo = train_grpo(model, prompts, reward_fn, epochs=3, group_size=4, max_new_tokens=4, lr=5e-4)

    metrics = {
        "base_final_loss": base.final_loss,
        "base_mean_loss": base.mean_loss,
        "sft_final_loss": sft.final_loss,
        "sft_mean_loss": sft.mean_loss,
        "dpo_final_loss": dpo.final_loss,
        "dpo_mean_loss": dpo.mean_loss,
        "grpo_final_loss": grpo.final_loss,
        "grpo_mean_loss": grpo.mean_loss,
        "sft_accuracy": evaluate_model(model, ins),
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }

    (RESULTS / "nanolm_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_demo()
