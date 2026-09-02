from __future__ import annotations

import json
from pathlib import Path

import torch

from nanolm.data import (
    build_instruction_pairs,
    build_pretrain_corpus,
    build_preference_pairs,
    train_tokenizer,
)
from nanolm.model import TinyDecoderLM
from nanolm.train import evaluate_model, sft_finetune, train_base_model, train_dpo

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def train_demo() -> None:
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

    metrics = {
        "base_final_loss": base.final_loss,
        "base_mean_loss": base.mean_loss,
        "sft_final_loss": sft.final_loss,
        "sft_mean_loss": sft.mean_loss,
        "dpo_final_loss": dpo.final_loss,
        "dpo_mean_loss": dpo.mean_loss,
        "sft_accuracy": evaluate_model(model, ins),
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }

    (RESULTS / "nanolm_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_demo()
