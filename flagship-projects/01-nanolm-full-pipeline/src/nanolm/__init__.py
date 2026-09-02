"""Minimal end-to-end decoder-only LLM pipeline for portfolio use."""

from nanolm.data import build_instruction_pairs, build_pretrain_corpus, build_preference_pairs
from nanolm.model import TinyDecoderLM
from nanolm.train import dpo_loss, evaluate_model, sft_finetune, train_base_model

__all__ = [
    "TinyDecoderLM",
    "build_instruction_pairs",
    "build_pretrain_corpus",
    "build_preference_pairs",
    "dpo_loss",
    "evaluate_model",
    "sft_finetune",
    "train_base_model",
]
__version__ = "0.1.0"
