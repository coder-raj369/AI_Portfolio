# MAINTENANCE

What in this repo is most likely to age badly, and what to do about it. A portfolio that
quotes a stale "current SOTA" is worse than one that never mentions it, so anything
time-sensitive is listed here with its expiry conditions.

Last reviewed: **2026-07-26**.

## Review cadence

| When | Do this |
|---|---|
| Every ~3 months | Re-run `make results` and `make smoke`; refresh any number that moved; bump the "last reviewed" date. |
| Every ~6 months | Re-check the "claims with a shelf life" table below; re-read the pinned-version table against current releases. |
| Before showing the repo to anyone | `make lint && make test && make smoke`. All three must be green. |

## Pinned versions and why

| Thing | Pinned to | Ages when |
|---|---|---|
| Python | 3.12 (floor) | 3.13/3.14 become the default in CI images. 3.12 was chosen over 3.10 because 3.10 reaches end-of-life in **October 2026**. |
| NumPy | ≥ 2.1 | NumPy 2.x already changed one behaviour this code relies on: `float(x)` on a size-1 **1-d** array now raises. If a future major release tightens more scalar conversions, the `.item()` calls in the test suites are the places to look. |
| PyTorch | 2.13 (target for M2+) | Not yet a dependency — no module in the tree needs it. The M2+ modules should pin whatever is current when they land, and prefer `torch.compile`-friendly code. |
| `regex` | ≥ 2024.9 | Only used for `\p{L}`-class pre-tokenization; very stable. |
| `uv` | any recent | `uv` replaced pip/poetry/pyenv as the default project manager during 2025-26; if that changes, `pyproject.toml` remains the portable artefact and only the `Makefile`/CI need editing. |
| `ruff` | ≥ 0.6 | Rule sets are renamed/expanded between minor versions; a lint failure after a bump usually means a rule moved, not that the code broke. |

## Claims with a shelf life

Statements in module READMEs that are true as of mid-2026 and should be re-checked:

1. **"RMSNorm is what every recent decoder-only LLM uses"**
   (`00-foundations/nn-primitives`). True across the Llama/Qwen/Gemma/Mistral families now.
   Re-check if a major release returns to LayerNorm or adopts something else (e.g. a
   normalisation-free residual scheme).
2. **"The 1.77× RMSNorm speedup does not transfer to GPU"** — measured on CPU NumPy. If this
   is ever re-measured with fused CUDA kernels, replace the number rather than the caveat.
3. **"GPT-4-style digit grouping (≤3) is probably wrong for arithmetic"**
   (`04-llms-and-genai/tokenizer-from-scratch`). Single-digit tokenization measurably helps
   maths and several 2025-26 models switched to it. If that becomes universal, the module
   should train both and report the difference instead of only noting it.
4. **"tiktoken's `cl100k_base` is the reference production vocabulary"** — newer OpenAI
   encodings (`o200k_base` and successors) are larger and compress better. The optional
   comparison test should be pointed at whatever is current.
5. **Optimiser landscape** — the module notes Shampoo/SOAP/Muon as the interesting frontier
   and does not implement them. If one of them becomes the default for open pretraining runs,
   that stops being a "nice follow-up" and starts being a gap.
6. **MCP spec version** — the M4 agent targets **MCP 2.1**. The spec is moving quickly;
   check the revision before quoting it.
7. **vLLM vs SGLang** — as of mid-2026 they are within run-to-run variance on raw throughput,
   with SGLang ahead on prefix-heavy workloads (RadixAttention) and TTFT. The M5 module must
   re-benchmark rather than repeat this, because it is exactly the kind of claim that inverts
   between releases.
8. **Model choices for M3–M5** (SmolLM2-135M/360M, Qwen2.5-1.5B) — chosen for Apache-2.0
   licensing and single-GPU fit. Newer small models appear constantly; the criteria (permissive
   license, ≤ 2 B params, fits ~16 GB with QLoRA) matter more than the specific checkpoint.

## Known weak spots, ranked

1. **No GPU-verified results yet.** Everything currently in the tree is CPU-only by design.
   The moment a PyTorch module lands, its README must carry real logs from a real run.
2. **Encode throughput in `bpetok` (~0.5 MB/s)** is 100-1000× off `tiktoken`. Documented in
   the module README; the fix is a compiled inner loop plus a pre-token cache.
3. **Small-scale-only conclusions.** Several findings (SGD+momentum beating Adam; schedules not
   helping) are true at ~1k parameters and should not be generalised. Each is captioned as
   such where it appears.
4. **No second-order or sharpness-aware optimiser**, no attention backward pass, no
   FlashAttention-style tiling. All three are listed as follow-ups in their module READMEs
   rather than silently missing.

## Things that will *not* age

Worth knowing which parts are safe: the autodiff engine, the finite-difference gradient
checks, the LayerNorm/RMSNorm derivations, the numerical-stability thresholds (they are
properties of IEEE-754, not of any library) and the BPE algorithm itself. Those are the parts
of the repo that stay true, which is a large part of why they were built first.
