"""Validate the NumPy reference implementations and record the measurements.

Everything here runs on CPU with NumPy only - no GPU, no downloads - and every
number in the "verified" half of this module's README comes from this script.
The PyTorch benchmarks live in `benchmark_torch.py` and require a GPU run.

Run: `python experiments/validate_reference.py`
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nanoformer.reference import (  # noqa: E402
    GroupedQueryAttention,
    KVCache,
    MoEFeedForward,
    SelectiveSSMBlock,
    apply_rope,
    load_balancing_loss,
    rope_frequencies,
    selective_scan_chunked,
    selective_scan_sequential,
    softplus,
)

RESULTS = Path(__file__).parent / "results"


def rope_properties() -> dict[str, object]:
    """RoPE's defining property, measured as a deviation rather than asserted."""
    rng = np.random.default_rng(0)
    cos, sin = rope_frequencies(64, 512)
    q, k = rng.standard_normal((1, 1, 1, 64)), rng.standard_normal((1, 1, 1, 64))

    def logit(m: int, n: int) -> float:
        return float(
            (apply_rope(q, cos, sin, offset=m) * apply_rope(k, cos, sin, offset=n)).sum()
        )

    # Same relative offset (+2) at wildly different absolute positions.
    logits = [logit(m, m - 2) for m in (2, 10, 100, 300, 500)]
    x = rng.standard_normal((2, 4, 32, 64))
    norms_before = np.linalg.norm(x, axis=-1)
    norms_after = np.linalg.norm(apply_rope(x, cos, sin, offset=97), axis=-1)
    return {
        "relative_position_logits": logits,
        "max_deviation_across_absolute_positions": float(np.ptp(logits)),
        "max_relative_deviation": float(np.ptp(logits) / abs(np.mean(logits))),
        "norm_preservation_max_abs_error": float(np.abs(norms_after - norms_before).max()),
    }


def kv_cache_equivalence() -> dict[str, object]:
    """Incremental decode vs one full forward, for three GQA configurations."""
    rng = np.random.default_rng(1)
    out: dict[str, object] = {}
    for n_kv in (8, 2, 1):
        attn = GroupedQueryAttention(64, n_heads=8, n_kv_heads=n_kv, max_seq=64, seed=0)
        x = rng.standard_normal((2, 16, 64))
        full = attn.forward(x, causal=True)
        cache = KVCache(2, attn.n_kv_heads, attn.head_dim, 64)
        steps = [attn.forward(x[:, t : t + 1], cache=cache, causal=True) for t in range(16)]
        incremental = np.concatenate(steps, axis=1)

        cache2 = KVCache(2, attn.n_kv_heads, attn.head_dim, 64)
        prefill = attn.forward(x[:, :10], cache=cache2, causal=True)
        decode = [
            attn.forward(x[:, t : t + 1], cache=cache2, causal=True) for t in range(10, 16)
        ]
        split = np.concatenate([prefill, *decode], axis=1)
        label = {8: "MHA (n_kv=8)", 2: "GQA (n_kv=2)", 1: "MQA (n_kv=1)"}[n_kv]
        out[label] = {
            "token_by_token_max_abs_error": float(np.abs(full - incremental).max()),
            "prefill_then_decode_max_abs_error": float(np.abs(full - split).max()),
        }
    return out


def kv_cache_memory_table() -> dict[str, object]:
    """Exact KV cache footprint at realistic scale - the reason GQA exists.

    Sized after a 7B-class model (32 layers, 32 heads, head_dim 128) at fp16,
    batch 1. These are closed-form byte counts, not timings, so they are exact
    rather than machine-dependent.
    """
    n_layers, n_heads, head_dim, dtype_bytes = 32, 32, 128, 2
    rows = []
    for label, n_kv in (("MHA (n_kv=32)", 32), ("GQA (n_kv=8)", 8), ("MQA (n_kv=1)", 1)):
        per_token = 2 * n_kv * head_dim * dtype_bytes * n_layers
        rows.append({
            "config": label,
            "n_kv_heads": n_kv,
            "bytes_per_token_all_layers": per_token,
            "gib_at_4k_context": round(per_token * 4_096 / 2**30, 3),
            "gib_at_32k_context": round(per_token * 32_768 / 2**30, 3),
            "gib_at_128k_context": round(per_token * 131_072 / 2**30, 3),
        })
    baseline = rows[0]["bytes_per_token_all_layers"]
    for r in rows:
        r["reduction_vs_mha"] = round(baseline / r["bytes_per_token_all_layers"], 1)
    return {
        "model_shape": f"{n_layers} layers, {n_heads} heads, head_dim {head_dim}, fp16, batch 1",
        "rows": rows,
    }


def ssm_vs_attention_state_cost() -> dict[str, object]:
    """The structural difference: attention state grows with sequence, SSM state does not."""
    n_layers, n_kv_heads, head_dim = 32, 8, 128
    d_inner, d_state = 8_192, 16  # comparable-width SSM layer
    attn_per_token = 2 * n_kv_heads * head_dim * 2 * n_layers
    ssm_state_bytes = d_inner * d_state * 4 * n_layers  # fp32 state, independent of seq
    rows = []
    for seq in (1_024, 8_192, 32_768, 131_072):
        rows.append({
            "sequence_length": seq,
            "attention_kv_cache_mib": round(attn_per_token * seq / 2**20, 1),
            "ssm_state_mib": round(ssm_state_bytes / 2**20, 1),
            "ratio": round(attn_per_token * seq / ssm_state_bytes, 2),
        })
    return {
        "note": "GQA-8 attention vs a d_inner=8192, d_state=16 SSM, 32 layers",
        "rows": rows,
    }


def moe_calibration() -> dict[str, object]:
    """Auxiliary loss across the full range from uniform to collapsed routing."""
    n_experts, tokens = 8, 1_024
    curve = []
    for skew in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        n_skewed = int(skew * tokens)
        idx = np.concatenate([
            np.zeros((n_skewed, 1), dtype=int),
            (np.arange(tokens - n_skewed) % n_experts).reshape(-1, 1),
        ])
        probs = np.zeros((tokens, n_experts))
        probs[np.arange(tokens), idx[:, 0]] = 1.0
        curve.append({
            "fraction_to_one_expert": skew,
            "load_balancing_loss": round(load_balancing_loss(probs, idx), 4),
        })

    uniform = np.full((tokens, n_experts), 1.0 / n_experts)
    uniform_idx = (np.arange(tokens * 2) % n_experts).reshape(tokens, 2)
    moe = MoEFeedForward(64, 128, n_experts=8, k=2, seed=0)
    _, init_stats = moe.forward(np.random.default_rng(2).standard_normal((8, 128, 64)))
    limited = MoEFeedForward(64, 128, n_experts=8, k=2, capacity_factor=0.25, seed=0)
    _, cap_stats = limited.forward(np.random.default_rng(2).standard_normal((8, 128, 64)))
    return {
        "loss_at_perfectly_uniform_routing": round(load_balancing_loss(uniform, uniform_idx), 6),
        "loss_at_total_collapse": round(
            load_balancing_loss(np.eye(n_experts)[np.zeros(tokens, int)],
                                np.zeros((tokens, 2), int)), 6),
        "n_experts": n_experts,
        "imbalance_curve": curve,
        "at_initialisation": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in init_stats.items()
        },
        "with_capacity_factor_0.25": {
            "tokens_dropped": cap_stats["tokens_dropped"],
            "tokens_total": 8 * 128 * 2,
            "dropped_fraction": round(cap_stats["tokens_dropped"] / (8 * 128 * 2), 4),
            "capacity_per_expert": cap_stats["capacity_per_expert"],
        },
    }


def ssm_scan_equivalence() -> dict[str, object]:
    """Chunked scan vs the sequential recurrence, across chunk sizes."""
    rng = np.random.default_rng(3)
    b, s, i, n = 2, 64, 8, 16
    x = rng.standard_normal((b, s, i))
    delta = softplus(rng.standard_normal((b, s, i)))
    A = -np.tile(np.arange(1, n + 1, dtype=np.float64), (i, 1))
    B, C, D = rng.standard_normal((b, s, n)), rng.standard_normal((b, s, n)), np.ones(i)

    start = time.perf_counter()
    sequential = selective_scan_sequential(x, delta, A, B, C, D)
    seq_ms = (time.perf_counter() - start) * 1000

    per_chunk = {}
    for chunk in (1, 4, 16, 32, 128):
        start = time.perf_counter()
        got = selective_scan_chunked(x, delta, A, B, C, D, chunk=chunk)
        per_chunk[str(chunk)] = {
            "max_abs_error_vs_sequential": float(np.abs(got - sequential).max()),
            "ms": round((time.perf_counter() - start) * 1000, 2),
        }

    with np.errstate(over="ignore", invalid="ignore"):
        long_x = rng.standard_normal((1, 200, i))
        long_delta = softplus(rng.standard_normal((1, 200, i)))
        long_B, long_C = rng.standard_normal((1, 200, n)), rng.standard_normal((1, 200, n))
        stable = selective_scan_sequential(long_x, long_delta, A, long_B, long_C, D)
        unstable = selective_scan_sequential(long_x, long_delta, -A, long_B, long_C, D)

    block = SelectiveSSMBlock(64, d_state=16, seed=0)
    u = rng.standard_normal((2, 48, 64))
    block_err = float(np.abs(block.forward(u) - block.forward(u, chunked=True)).max())

    return {
        "shape": {"batch": b, "seq": s, "d_inner": i, "d_state": n},
        "sequential_ms": round(seq_ms, 2),
        "per_chunk_size": per_chunk,
        "full_block_max_abs_error": block_err,
        "stability": {
            "seq_len": 200,
            "negative_A_max_abs_output": round(float(np.abs(stable).max()), 4),
            "negative_A_all_finite": bool(np.all(np.isfinite(stable))),
            "positive_A_all_finite": bool(np.all(np.isfinite(unstable))),
        },
    }


def plot(moe: dict, kv_mem: dict, ssm_cost: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 3.6), dpi=140)

    xs = [p["fraction_to_one_expert"] for p in moe["imbalance_curve"]]
    ys = [p["load_balancing_loss"] for p in moe["imbalance_curve"]]
    ax1.plot(xs, ys, "o-", lw=1.7)
    ax1.axhline(1.0, ls="--", c="grey", lw=1)
    ax1.text(0.02, 1.15, "minimum = 1.0 (uniform)", fontsize=7, color="grey")
    ax1.set(xlabel="fraction of tokens sent to one expert",
            ylabel="load-balancing loss", title="MoE aux loss is calibrated")

    labels = [r["config"] for r in kv_mem["rows"]]
    gib = [r["gib_at_32k_context"] for r in kv_mem["rows"]]
    ax2.bar(range(len(labels)), gib, color=["#d93025", "#f9ab00", "#1e8e3e"])
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set(ylabel="GiB", title="KV cache at 32k context (7B-class, fp16)")
    for i, g in enumerate(gib):
        ax2.text(i, g, f"{g:.2f}", ha="center", va="bottom", fontsize=7.5)

    seqs = [r["sequence_length"] for r in ssm_cost["rows"]]
    ax3.plot(seqs, [r["attention_kv_cache_mib"] for r in ssm_cost["rows"]], "o-",
             lw=1.7, label="attention KV cache")
    ax3.plot(seqs, [r["ssm_state_mib"] for r in ssm_cost["rows"]], "s-",
             lw=1.7, label="SSM state")
    ax3.set(xlabel="sequence length", ylabel="MiB", title="Decode state vs context length")
    ax3.set_xscale("log", base=2)
    ax3.set_yscale("log")
    ax3.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "reference_validation.png")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": "NumPy float64, CPU only - no GPU or downloads required",
        "rope": rope_properties(),
        "kv_cache_equivalence": kv_cache_equivalence(),
        "kv_cache_memory": kv_cache_memory_table(),
        "ssm_vs_attention_decode_state": ssm_vs_attention_state_cost(),
        "moe_routing": moe_calibration(),
        "ssm_scan": ssm_scan_equivalence(),
    }
    plot(payload["moe_routing"], payload["kv_cache_memory"],
         payload["ssm_vs_attention_decode_state"])
    (RESULTS / "reference_validation.json").write_text(json.dumps(payload, indent=2) + "\n")

    r = payload["rope"]
    print(f"RoPE relative-position deviation : {r['max_relative_deviation']:.2e} (relative)")
    print(f"RoPE norm preservation error     : {r['norm_preservation_max_abs_error']:.2e}")
    for name, v in payload["kv_cache_equivalence"].items():
        print(f"KV cache {name:14s}      : {v['token_by_token_max_abs_error']:.2e} "
              f"(token-by-token), {v['prefill_then_decode_max_abs_error']:.2e} (prefill+decode)")
    m = payload["moe_routing"]
    print(f"MoE aux loss uniform / collapsed : {m['loss_at_perfectly_uniform_routing']} / "
          f"{m['loss_at_total_collapse']}")
    print(f"MoE dropped at capacity 0.25     : {m['with_capacity_factor_0.25']['dropped_fraction']:.1%}")
    s = payload["ssm_scan"]
    worst = max(v["max_abs_error_vs_sequential"] for v in s["per_chunk_size"].values())
    print(f"SSM chunked vs sequential (worst): {worst:.2e}")
    print(f"SSM block seq vs chunked         : {s['full_block_max_abs_error']:.2e}")
    print(f"SSM positive-A finite at seq 200 : {s['stability']['positive_A_all_finite']}")
    for row in payload["kv_cache_memory"]["rows"]:
        print(f"{row['config']:15s} {row['gib_at_32k_context']:6.2f} GiB @32k  "
              f"({row['reduction_vs_mha']}x vs MHA)")


if __name__ == "__main__":
    main()
