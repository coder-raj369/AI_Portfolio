# nanoLM full pipeline

## What it is

This flagship project is a compact decoder-only language model pipeline built from first principles for a consumer GPU / CPU budget. The stack is intentionally small but complete: a toy BPE tokenizer, a minimal decoder-only transformer, a pretraining stage, a supervised fine-tuning stage, and a preference-alignment stage using a DPO-style objective.

The goal is not to train a frontier model. The goal is to prove the full engineering pipeline end to end on a tiny but meaningful scale, with the code and metrics clearly documented.

## Why it matters

A hiring team wants to see the full LLM stack working, not just a prompt wrapper around a model API. This module demonstrates:

- tokenization and synthetic corpus setup
- causal language modeling with positional embeddings
- training loops for base pretraining and SFT
- reward-preference alignment with DPO-style objective
- a public eval story with real measured outcomes

## Architecture

```mermaid
flowchart TD
    A[Corpus] --> B[Tokenizer]
    B --> C[Pretraining batches]
    C --> D[Decoder-only model]
    D --> E[Base eval]
    E --> F[SFT on instruction pairs]
    F --> G[Instruction eval]
    G --> H[DPO on preference pairs]
    H --> I[Aligned eval]
```

## Run it

```bash
cd flagship-projects/01-nanolm-full-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m pytest -q
python3 experiments/train_pipeline.py
```

## Estimated behavior

This is intentionally a small model and the main objective is pipeline correctness, not state-of-the-art quality. On a small synthetic corpus and a compact instruction set, the model learns to reproduce the obvious pattern structure quickly and can be evaluated with exact-match-style checks.

## Current status

This module is intentionally scoped to CPU-safe learning over a synthetic corpus and tiny instruction set. It is designed to run in a local environment without external model downloads and without large artifacts.

## Model card

- Intended use: education, architecture validation, and portfolio demonstration.
- Training data: synthetic corpus generated from a small set of common facts and patterns.
- Limitations: toy scale, narrow distribution, no real-world generalization claim.
- License: MIT (repo root).
