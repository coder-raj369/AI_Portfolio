# PROGRESS

Live build state for `ai-portfolio`. A session with no memory of prior work should be able to
read this file and resume without re-deriving the plan.

**Current state:** M0 (scaffold) and M1 (foundations + tokenizer) are **complete, tested and
documented**. The M2 evaluation substrate is complete, and M3 nanoLM plus the M4 codebase
copilot foundation are in active build. New flagship work remains uncommitted until explicitly
requested.
**Last updated:** 2026-07-26.

## Ground rules (do not drift from these)
- Implement → test → document → record real results → commit. Stop only at clean module
  boundaries, never mid-file.
- **No number in any README that did not come from a real run log.** If a run has not happened,
  the table cell says `RUN PENDING` — never an estimate.
- Every project folder: `README.md` (Mermaid diagram + quantified results + "what didn't work"),
  `pyproject.toml`, typed/documented `src/<pkg>/`, `tests/`, and `experiments/results/*.json`
  where it produces numbers.
- Model weights are never committed; download from the Hugging Face Hub in code + ship a model
  card. Permissive licenses only (Apache-2.0 / MIT / ODC-By), noted per module README.
- Python 3.12, `uv` + `uv.lock`, `ruff` (lint+format, 100 cols), `pytest`, Docker for
  flagships, GitHub Actions CI for lint + CPU tests.
- **Authoring constraint:** the environment these modules were written in has no GPU, no
  network and no `torch`. Everything in M0/M1 is therefore NumPy-only and fully executed. For
  M2+ (PyTorch), code ships with a `make results` runner and tables marked `RUN PENDING` until
  a real GPU log exists.

## Decisions log
| Date | Decision |
|---|---|
| 2026-07-26 | Broad four-pillar scope; PyTorch primary (JAX only as a scaling-law side note); single ~16 GB GPU / Colab T4 budget; MIT license; Apache-2.0 base models only. |
| 2026-07-26 | Speech module dropped (lowest signal-per-hour); GNN + time-series added as `08-specialized-domains`. |
| 2026-07-26 | 4th flagship (DPO reproduction) lives in `06-research-reproductions/dpo-small-scale`, linked from the root README rather than duplicated under `flagship-projects/`. |
| 2026-07-26 | `04-llms-and-genai/interpretability-and-safety` added (absent from the original taxonomy). |
| 2026-07-26 | SSM (Mamba-style) + MoE live inside `transformers-from-scratch` as sibling blocks, not as separate thin folders. |
| 2026-07-26 | Root `pyproject.toml` drives ruff+pytest repo-wide (single CI entrypoint) while each project keeps its own `pyproject.toml` to stay standalone-installable. |
| 2026-07-26 | Tokenizer corpus = the repo's own text (MIT, offline, zero licensing risk); `--input` flag provided for FineWeb-Edu when natural-language numbers are wanted. |

## Milestones
- [x] **M0 — Repo skeleton.** MIT license, `.gitignore`, root `pyproject.toml` (ruff + pytest
      pythonpath), `Makefile` (setup/lint/format/test/smoke/results), `.github/workflows/ci.yml`,
      `scripts/smoke_test.py` (behavioural, not import-only), root `README.md`, `MAINTENANCE.md`.
- [x] **M1 — Foundations + tokenizer.**
  - [x] `00-foundations/micrograd-engine` — nanograd autodiff, 47 tests, gradcheck 1.4e-09.
  - [x] `00-foundations/optimizers-from-scratch` — nanoptim, 30 tests, AdamW decoupling proven.
  - [x] `00-foundations/nn-primitives` — nnprim, 44 tests, overflow thresholds measured.
  - [x] `04-llms-and-genai/tokenizer-from-scratch` — bpetok, 49 tests, 3.25 held-out B/tok.
- [ ] **M2 — Architecture substrate.** `02-deep-learning/transformers-from-scratch` (MHA from
      scratch in torch, RoPE/ALiBi, RMSNorm, GQA, KV cache, MoE with load-balancing loss,
      Mamba-style selective scan + hybrid block, online-softmax attention);
      `04-llms-and-genai/evaluation-harness` (task registry, pass@k, verifier scoring,
      LLM-judge + human-agreement spot check, regression gating).
- [ ] **M3 — Flagship 1: nanoLM full pipeline.** ~50 M-param decoder: pretrain on a
  FineWeb-Edu slice → SFT → DPO → GRPO/RLVR with a programmatic verifier; base/SFT/DPO/GRPO
  eval table incl. alignment tax; best-of-n + self-consistency at matched compute. The
  current CPU-safe implementation covers all four training stages on synthetic data.
  Feeds `03-reinforcement-learning/llm-alignment` and `reasoning-rl`.
- [ ] **M4 — Flagship 2: codebase copilot.** AST-aware chunking, dense+BM25 hybrid,
      cross-encoder rerank, MCP 2.1 server (`search`/`read_file`/`propose_patch`), planning
      agent with failure taxonomy, FastAPI + UI, Docker Compose, 60-question eval set +
      ablation grid. Feeds `rag-pipeline` and `llm-agents`.
- [ ] **M5 — Flagship 3 + paper reproductions.** `efficient-inference-server` (naive → KV cache
      → static → continuous batching → INT4/AWQ → vLLM/SGLang; p50/p95/TTFT/ITL, cost per 1 M
      tokens); `06-research-reproductions/dpo-small-scale` (+ β vs length-bias ablation + honest
      gap section); `ddpm-ddim-sampler`.
- [ ] **M6 — Breadth fill.** `01-classical-ml`, `02-deep-learning/{cnns,vision-transformers,
      generative-models,multimodal-clip}`, `03-reinforcement-learning/{classical-rl,deep-rl}`,
      `04-.../interpretability-and-safety`, `05-mlops-and-systems`, `07-system-design-docs`
      (5 staff-level docs), `08-specialized-domains/{graph-neural-networks,time-series}`.
- [ ] **M7 — Integration pass.** Style consistency, root README polish, full smoke test,
      results audit (zero `RUN PENDING` left), portfolio blurb, push instructions.

## Results ledger (every entry traceable to a committed log)
| Module | Metric | Value | Source |
|---|---|---|---|
| micrograd-engine | worst gradcheck rel. error, 17 expressions | 1.4e-09 | `experiments/results/gradcheck_errors.json` |
| micrograd-engine | spirals train / test accuracy | 99.3% / 95.6% | `experiments/results/metrics.json` |
| micrograd-engine | sklearn L-BFGS baseline (same task) | 100% / 94.1% | same |
| micrograd-engine | throughput | 2,802 steps/s CPU (0.54 s / 1,500 steps) | same |
| optimizers | quadratic: steps to 1e-6× initial loss | Nesterov 86 · momentum 123 · AdamW 123 · Adam 129 · SGD 311 · Lion never | `results/convex_benchmark.json` |
| optimizers | spirals test accuracy (per-optimiser LR sweep) | SGD+mom 97.0% · Lion 95.6% · Nesterov 96.3% · Adam 94.8% · AdamW 94.8% · SGD 80.0% | `results/spirals_benchmark.json` |
| optimizers | AdamW vs Adam+L2 step at zero gradient | 0.01 vs 0.10 (10× ), scale-invariance to 1e-9 | `tests/test_adamw_decoupling.py` |
| optimizers | cosine+warmup on best config | 97.0% → 96.3% (**hurt**) | `results/spirals_benchmark.json` |
| nn-primitives | `exp` overflow threshold | fp16 11.09 · fp32 88.72 · fp64 709.78 | `results/stability_report.json` |
| nn-primitives | naive vs stable log-softmax @ logit 90 (fp32) | `nan` vs 0.0 error | same |
| nn-primitives | hand-derived gradients, worst abs error | 8.4e-10 (LN dx) | same |
| nn-primitives | RMSNorm vs LayerNorm forward, 32×512×1024 | 150.8 ms vs 267.3 ms (1.77×) | same |
| nn-primitives | attention entropy with / without 1/√d_k | 2.98 / 0.54 nats (max weight 0.99996) | same |
| nn-primitives | MHA vs naive per-head loop | 1.1e-16 | `tests/test_attention.py` |
| tokenizer | held-out bytes/token @ 4,552 vocab | **3.246** (in-sample 3.644) | `experiments/results/benchmark.json` |
| tokenizer | vocab requested 8,192 → actual | 4,552 (corpus exhausted; reported, not padded) | same |
| tokenizer | encode throughput | 0.48 MB/s (100-1000× slower than tiktoken — known) | same |
| repo | test suite | 170 passed, 3 skipped (optional torch/tiktoken) | `make test` |
| repo | smoke test | 4/4 modules healthy | `make smoke` |

## Open questions / blockers
1. **GPU verification path for M2+.** Default assumption: code ships runnable with `make
   results`; the user executes on their GPU/Colab and returns logs; CI verifies the CPU test
   suite. Confirmed as the working plan for this build.
2. **Repo name / GitHub username** not supplied — README badges are placeholder-linked (`#`)
   and should be pointed at the real repo before publishing.
3. **`ruff` was not installed in the authoring environment.** Run `make format && make lint`
   once after cloning; CI enforces it on push.

## How to push this to GitHub
```bash
cd ai-portfolio
git log --oneline                 # 6 incremental commits already exist
git remote add origin git@github.com:<username>/<repo>.git
git branch -M main && git push -u origin main
```
Commits are authored as `rajucoding <rajucoding@users.noreply.github.com>`; rewrite with
`git rebase -i --exec 'git commit --amend --no-edit --reset-author'` if a different identity is
wanted.
