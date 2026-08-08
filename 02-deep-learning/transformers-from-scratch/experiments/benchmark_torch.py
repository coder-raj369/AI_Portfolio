"""PyTorch benchmarks for the blocks in this module. **Requires a GPU run.**

The authoring environment for this repo has no GPU, so the result tables in the
README marked `RUN PENDING` are filled from this script's output rather than
estimated. It writes `results/torch_benchmark.json`; the README consumes that file.

What it measures:
  1. Our hand-written attention vs `F.scaled_dot_product_attention` (which dispatches
     to a fused/flash kernel) - latency and peak memory across sequence lengths.
  2. Decode throughput with and without a KV cache, to quantify the O(n^2) -> O(n)
     difference rather than assert it.
  3. GQA sweep: quality-neutral memory reduction from shrinking `n_kv_heads`.
  4. SSM sequential vs chunked scan, and SSM vs attention scaling in sequence length.
  5. A dense-FFN vs MoE parameter/latency comparison at matched active parameters.

Run:
    python experiments/benchmark_torch.py                 # auto-detects CUDA
    python experiments/benchmark_torch.py --device cpu    # smaller shapes, still works
    python experiments/benchmark_torch.py --quick         # ~1 min instead of ~10
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from nanoformer.torch_impl import (
    GroupedQueryAttention,
    KVCache,
    ModelConfig,
    MoEFeedForward,
    SelectiveSSM,
    SwiGLUFeedForward,
    TinyDecoderLM,
    selective_scan_chunked,
    selective_scan_sequential,
)

RESULTS = Path(__file__).parent / "results"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed(fn, device: str, warmup: int = 3, iters: int = 10) -> float:
    """Median-of-`iters` wall time in milliseconds, after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return round(samples[len(samples) // 2], 3)


def peak_mib(device: str) -> float:
    if device != "cuda":
        return float("nan")
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def reset_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def bench_attention_vs_sdpa(device: str, seq_lens: list[int], dtype) -> list[dict]:
    """Our raw-ops attention vs the fused library kernel.

    The gap is the value of a fused kernel: same mathematics, no materialised
    (T, T) score matrix, so memory stays linear in sequence length.
    """
    d_model, n_heads = 512, 8
    rows = []
    for seq in seq_lens:
        attn = GroupedQueryAttention(d_model, n_heads, n_heads, max_seq=max(seq_lens)).to(
            device=device, dtype=dtype
        )
        x = torch.randn(2, seq, d_model, device=device, dtype=dtype)

        reset_peak(device)
        with torch.no_grad():
            ours_ms = timed(lambda: attn(x, causal=True), device)
        ours_mem = peak_mib(device)

        def fused() -> torch.Tensor:
            b, s, _ = x.shape
            hd = d_model // n_heads
            q = attn.w_q(x).view(b, s, n_heads, hd).transpose(1, 2)
            k = attn.w_k(x).view(b, s, n_heads, hd).transpose(1, 2)
            v = attn.w_v(x).view(b, s, n_heads, hd).transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            return attn.w_o(out.transpose(1, 2).reshape(b, s, d_model))

        reset_peak(device)
        with torch.no_grad():
            fused_ms = timed(fused, device)
        fused_mem = peak_mib(device)

        rows.append({
            "seq_len": seq,
            "ours_ms": ours_ms,
            "sdpa_ms": fused_ms,
            "speedup_of_sdpa": round(ours_ms / fused_ms, 2) if fused_ms else None,
            "ours_peak_mib": ours_mem,
            "sdpa_peak_mib": fused_mem,
        })
    return rows


def bench_kv_cache(device: str, n_new: int, dtype) -> dict:
    """Decode with a KV cache vs recomputing the prefix every step."""
    cfg = ModelConfig(vocab_size=1024, d_model=512, n_layers=6, n_heads=8, n_kv_heads=2,
                      max_seq=1024, layer_pattern=("attn",) * 6)
    model = TinyDecoderLM(cfg).to(device=device, dtype=dtype).eval()
    prompt = torch.randint(0, 1024, (1, 64), device=device)

    def cached() -> None:
        caches = [
            KVCache(1, b.mixer.n_kv_heads, b.mixer.head_dim, cfg.max_seq,
                    device=device, dtype=dtype)
            for b in model.blocks
        ]
        logits, _ = model(prompt, caches=caches)
        tokens = prompt
        for _ in range(n_new):
            nxt = logits[:, -1:].argmax(dim=-1)
            tokens = torch.cat([tokens, nxt], dim=1)
            logits, _ = model(nxt, caches=caches)

    def uncached() -> None:
        tokens = prompt
        for _ in range(n_new):
            logits, _ = model(tokens)
            tokens = torch.cat([tokens, logits[:, -1:].argmax(dim=-1)], dim=1)

    with torch.no_grad():
        cached_ms = timed(cached, device, warmup=1, iters=3)
        uncached_ms = timed(uncached, device, warmup=1, iters=3)
    return {
        "prompt_tokens": 64,
        "generated_tokens": n_new,
        "with_kv_cache_ms": cached_ms,
        "without_kv_cache_ms": uncached_ms,
        "speedup": round(uncached_ms / cached_ms, 2) if cached_ms else None,
        "cached_tokens_per_second": round(n_new / (cached_ms / 1000), 1) if cached_ms else None,
    }


def bench_gqa(device: str, dtype) -> list[dict]:
    d_model, n_heads, seq = 512, 8, 512
    rows = []
    for n_kv in (8, 4, 2, 1):
        attn = GroupedQueryAttention(d_model, n_heads, n_kv, max_seq=seq).to(
            device=device, dtype=dtype
        )
        x = torch.randn(2, seq, d_model, device=device, dtype=dtype)
        with torch.no_grad():
            ms = timed(lambda: attn(x, causal=True), device)
        rows.append({
            "n_kv_heads": n_kv,
            "forward_ms": ms,
            "kv_bytes_per_token": attn.kv_bytes_per_token(2),
            "params": sum(p.numel() for p in attn.parameters()),
        })
    return rows


def bench_ssm(device: str, seq_lens: list[int], dtype) -> dict:
    d_model = 256
    ssm = SelectiveSSM(d_model, d_state=16).to(device=device, dtype=dtype)
    attn = GroupedQueryAttention(d_model, 8, 2, max_seq=max(seq_lens)).to(
        device=device, dtype=dtype
    )
    rows = []
    for seq in seq_lens:
        x = torch.randn(1, seq, d_model, device=device, dtype=dtype)
        with torch.no_grad():
            seq_ms = timed(lambda: ssm(x, chunked=False), device, warmup=1, iters=3)
            chunk_ms = timed(lambda: ssm(x, chunked=True), device, warmup=1, iters=3)
            attn_ms = timed(lambda: attn(x, causal=True), device, warmup=1, iters=3)
            equal = torch.allclose(
                ssm(x, chunked=False), ssm(x, chunked=True), atol=1e-4
            )
        rows.append({
            "seq_len": seq,
            "ssm_sequential_ms": seq_ms,
            "ssm_chunked_ms": chunk_ms,
            "attention_ms": attn_ms,
            "scan_paths_agree": bool(equal),
        })
    return {"note": "SSM d_model=256 d_state=16 vs GQA-2 attention, same d_model", "rows": rows}


def bench_moe(device: str, dtype) -> dict:
    d_model, d_hidden, seq = 512, 1024, 256
    dense = SwiGLUFeedForward(d_model, d_hidden).to(device=device, dtype=dtype)
    moe = MoEFeedForward(d_model, d_hidden, n_experts=8, k=2).to(device=device, dtype=dtype)
    x = torch.randn(4, seq, d_model, device=device, dtype=dtype)
    with torch.no_grad():
        dense_ms = timed(lambda: dense(x), device)
        moe_ms = timed(lambda: moe(x)[0], device)
        _, stats = moe(x)
    dense_params = sum(p.numel() for p in dense.parameters())
    moe_params = sum(p.numel() for p in moe.parameters())
    per_expert = sum(p.numel() for p in moe.experts[0].parameters())
    return {
        "dense_ffn_ms": dense_ms,
        "moe_ms": moe_ms,
        "dense_params": dense_params,
        "moe_total_params": moe_params,
        "moe_active_params_per_token": 2 * per_expert,
        "capacity_multiplier": round(moe_params / dense_params, 2),
        "load_balancing_loss_at_init": round(float(stats.load_balancing_loss), 4),
        "dead_experts_at_init": stats.dead_experts,
        "imbalance_ratio_at_init": round(stats.imbalance_ratio, 3),
        "note": (
            "The Python for-loop over experts makes this MoE far slower than the dense "
            "FFN at these shapes. A real implementation uses grouped GEMM / expert "
            "parallelism; the loop is here for legibility and the gap is the honest "
            "cost of that choice."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="smaller shapes, ~1 minute")
    args = parser.parse_args()
    device = args.device
    dtype = torch.float16 if device == "cuda" else torch.float32

    seq_lens = [128, 512] if args.quick else [128, 512, 1024, 2048]
    ssm_lens = [128, 512] if args.quick else [128, 512, 1024]
    n_new = 32 if args.quick else 128

    print(f"device={device} dtype={dtype} torch={torch.__version__}")
    payload = {
        "environment": {
            "torch": torch.__version__,
            "device": device,
            "dtype": str(dtype),
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
            "platform": platform.platform(),
            "quick_mode": args.quick,
        },
        "attention_vs_sdpa": bench_attention_vs_sdpa(device, seq_lens, dtype),
        "kv_cache": bench_kv_cache(device, n_new, dtype),
        "gqa_sweep": bench_gqa(device, dtype),
        "ssm": bench_ssm(device, ssm_lens, dtype),
        "moe": bench_moe(device, dtype),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "torch_benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {RESULTS / 'torch_benchmark.json'}")


if __name__ == "__main__":
    main()
