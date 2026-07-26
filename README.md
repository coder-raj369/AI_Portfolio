# AI Portfolio — ML · DL · RL · LLM engineering, built from first principles

A working portfolio across classical ML, deep learning, reinforcement learning and modern
LLM/generative-AI engineering. Core building blocks are implemented **from scratch and
verified numerically**, and every result quoted anywhere in this repo comes from a run log
committed alongside it.

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.12-blue" alt="python 3.12" /></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-170%20passing-brightgreen" alt="tests" /></a>
  <a href="#"><img src="https://img.shields.io/badge/lint-ruff-000000" alt="ruff" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT" /></a>
</p>

> **Status: milestones M0–M1 complete and running.** 4 modules, 170 passing tests, 100% of
> quoted numbers reproducible from committed logs. The remaining modules are planned in
> [`PROGRESS.md`](PROGRESS.md) and are **not** in the tree yet — nothing here is a stub or a
> placeholder. See [Roadmap](#roadmap).

## If you only look at three things

| # | Look at | Why |
|---|---|---|
| 1 | [`00-foundations/micrograd-engine`](00-foundations/micrograd-engine) | A reverse-mode autodiff engine in NumPy, **gradient-checked to 1.4e-09** against finite differences across 17 expressions, then trained to 95.6% test accuracy on a non-linear task — matching a strong sklearn L-BFGS baseline (94.1%). |
| 2 | [`00-foundations/nn-primitives`](00-foundations/nn-primitives) | The numerical guts of a transformer: **measured** overflow thresholds (fp16 **11.09**, fp32 **88.72**, fp64 **709.78**), hand-derived LayerNorm/RMSNorm backward passes verified to 8.4e-10, and attention matching a naive per-head loop to **1.1e-16** with causality proven by perturbation. |
| 3 | [`04-llms-and-genai/tokenizer-from-scratch`](04-llms-and-genai/tokenizer-from-scratch) | Byte-level BPE end to end: **3.25 held-out bytes/token**, 1,000-case Unicode fuzz roundtrip, merge-order semantics, and prompt-injection-safe special-token handling. |

Each of those three has a README with an architecture diagram, quantified results, and an
explicit **"what didn't work / limitations"** section. Three findings that did not flatter the
implementation and are reported anyway:

- Tuned **SGD+momentum beats tuned Adam** on the small net (97.0% vs 94.8%) — adaptive methods
  are not universally better, and the benchmark says so.
- **Lion never converges at a fixed learning rate** on either convex objective; its
  unit-magnitude update can only orbit the optimum without a decay schedule.
- Adding a **cosine+warmup schedule made the best configuration slightly worse** (97.0% →
  96.3%). The ablation that didn't pay off is still in the table.

## Contents

| Module | What it contains | Status |
|---|---|---|
| [`00-foundations/micrograd-engine`](00-foundations/micrograd-engine) | Reverse-mode autodiff `Tensor` (20 broadcasting-correct ops), iterative backward pass, stable `log_softmax`/cross-entropy, minimal module system, finite-difference gradchecker | ✅ 47 tests |
| [`00-foundations/optimizers-from-scratch`](00-foundations/optimizers-from-scratch) | SGD (+momentum/Nesterov), Adam, AdamW, Lion, `clip_grad_norm_`, cosine-with-warmup and warmup-stable-decay schedules; Adam+L2 vs AdamW divergence proven numerically | ✅ 30 tests |
| [`00-foundations/nn-primitives`](00-foundations/nn-primitives) | Stable softmax/logsumexp/cross-entropy, LayerNorm + RMSNorm forward **and** backward, multi-head attention from raw ops, Xavier/Kaiming/truncated/GPT-2-residual init | ✅ 44 tests |
| [`04-llms-and-genai/tokenizer-from-scratch`](04-llms-and-genai/tokenizer-from-scratch) | Byte-level BPE trainer with an inverted pair index, GPT-4 regex pre-tokenization, special-token policies, JSON round-trip, vocab-size ablation | ✅ 49 tests + 2 optional |
| `02-deep-learning/transformers-from-scratch` | MHA/RoPE/ALiBi/RMSNorm/GQA/KV-cache, Mixture-of-Experts, Mamba-style selective SSM + hybrid block | 🔜 M2 |
| `04-llms-and-genai/evaluation-harness` | Shared eval infra: task registry, pass@k, verifier scoring, LLM-judge with rubric, regression gating | 🔜 M2 |
| `flagship-projects/01-nanolm-full-pipeline` | tokenizer → pretrain → SFT → DPO → GRPO/RLVR with a base/SFT/DPO/GRPO comparison table | 🔜 M3 |
| `flagship-projects/02-codebase-copilot` | Hybrid retrieval + re-ranker, MCP 2.1 agent, FastAPI + Docker, 60-question eval set | 🔜 M4 |
| `flagship-projects/03-efficient-inference-server` | KV-cache → continuous batching → INT4/AWQ → vLLM/SGLang, latency/throughput/cost | 🔜 M5 |
| `06-research-reproductions` | DPO at small scale + an ablation the paper didn't run; DDPM→DDIM sampler | 🔜 M5 |
| `01`, `02`, `03`, `05`, `07`, `08` | Classical ML, CNNs/ViT/generative/CLIP, RL (tabular→PPO→RLHF), MLOps, 5 system-design docs, GNNs + time series | 🔜 M6 |

Full plan, per-module descriptions and build order: [`PROGRESS.md`](PROGRESS.md).

## The four things this repo is trying to demonstrate

1. **Fundamentals** — autodiff, optimisers, normalisation backward passes, attention and BPE
   written out and *verified*, not described.
2. **Applied engineering** — pinned per-project environments, 170 tests, CI that lints and
   tests on push, a smoke test that exercises every module in under 5 seconds.
3. **Current relevance** — RMSNorm/SwiGLU/GQA, WSD schedules, GRPO/RLVR, MCP-wired agents,
   quantisation and continuous batching (M2–M5).
4. **Research capability** — measured ablations, honest negative results, paper reproductions
   with a "how close did we get" section, and staff-level system-design docs.

## Setup

```bash
git clone <this repo> && cd ai-portfolio
uv sync                      # Python 3.12 + pinned dev deps (uv installs the interpreter too)

make test                    # 170 tests, CPU only, ~25 s
make smoke                   # imports + exercises every module, ~5 s
make results                 # regenerates every number quoted in every README, ~60 s
make lint                    # ruff check + format check (what CI runs)
```

No GPU, no downloads and no API keys are needed for anything currently in the tree — all
datasets are generated procedurally or derived from the repo's own text. Modules that *do*
need a GPU (M3 onwards) will state their compute budget in their own README; everything is
scoped to a single ~16 GB consumer GPU or a free-tier Colab T4.

Each project is also standalone:

```bash
cd 00-foundations/micrograd-engine
pip install -e ".[experiments,dev]" && pytest && python experiments/train_spirals.py
```

## Repository conventions

- `NN-kebab-case/` numbered sections; every project folder has `README.md`,
  `pyproject.toml`, `src/<package>/`, `tests/` and (where it produces numbers) `experiments/`.
- **Every README result is backed by a committed `results/*.json` and the script that made
  it.** Unverified numbers are never written down — a table cell either holds a real measured
  value or says so.
- Type hints and docstrings throughout; `ruff` (lint + format) with a 100-column limit.
- Model weights are never committed. Anything trained will be downloaded in code from the
  Hugging Face Hub and shipped with a short model card.
- Datasets and base models are permissively licensed (Apache-2.0 / MIT / ODC-By), with the
  license noted in the module README, because this repo is public.

## Roadmap

`M0` scaffold + CI · `M1` foundations + tokenizer ✅ → `M2` transformer/SSM/MoE substrate +
eval harness → `M3` nanoLM full pipeline (pretrain → SFT → DPO → GRPO) → `M4` codebase copilot
(hybrid RAG + MCP agent) → `M5` inference server + paper reproductions → `M6` breadth fill and
system-design docs → `M7` integration pass.

[`PROGRESS.md`](PROGRESS.md) is kept current and is the source of truth for what is done.
[`MAINTENANCE.md`](MAINTENANCE.md) tracks what will age fastest.

## License

[MIT](LICENSE).
