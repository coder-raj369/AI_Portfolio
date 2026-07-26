"""Measure where the naive formulas break, and verify the hand-derived gradients.

Everything quoted in this module's README is produced here. Nothing in this file
depends on a GPU or a download.

Run: `python experiments/stability_report.py`
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nnprim.attention import scaled_dot_product_attention  # noqa: E402
from nnprim.functional import (  # noqa: E402
    gelu,
    log_softmax,
    naive_log_softmax,
    naive_softmax,
    softmax,
    softmax_backward,
)
from nnprim.gradcheck import max_abs_error, numeric_grad  # noqa: E402
from nnprim.norm import layer_norm, layer_norm_backward, rms_norm, rms_norm_backward  # noqa: E402

RESULTS = Path(__file__).parent / "results"
DTYPES = {"float16": np.float16, "float32": np.float32, "float64": np.float64}


def exp_overflow_threshold(dtype) -> float:
    """Bisect the smallest x for which `exp(x)` is not finite in `dtype`."""
    lo, hi = 0.0, 1e5
    with np.errstate(over="ignore"):
        for _ in range(200):
            mid = (lo + hi) / 2
            if np.isfinite(np.exp(np.array(mid, dtype=dtype))):
                lo = mid
            else:
                hi = mid
    return lo


def first_naive_softmax_failure(dtype) -> float:
    """Smallest logit magnitude at which the naive softmax stops being finite."""
    with np.errstate(over="ignore", invalid="ignore"):
        for magnitude in range(0, 100000):
            x = np.array([[magnitude, magnitude - 1.0, magnitude - 2.0]], dtype=dtype)
            if not np.all(np.isfinite(naive_softmax(x))):
                return float(magnitude)
    return float("inf")


def dtype_table() -> dict[str, dict[str, float]]:
    out = {}
    for name, dtype in DTYPES.items():
        thresh = exp_overflow_threshold(dtype)
        fail = first_naive_softmax_failure(dtype)
        x = np.array([[fail + 10, fail + 9, fail + 8]], dtype=dtype)
        stable_ok = bool(np.all(np.isfinite(softmax(x.astype(np.float64)))))
        out[name] = {
            "exp_overflow_at": round(thresh, 3),
            "naive_softmax_first_nan_at_logit": fail,
            "stable_softmax_finite_past_failure": stable_ok,
        }
    return out


def error_curve() -> dict[str, list[float]]:
    """Max absolute error of naive vs stable log-softmax as logits grow (float32)."""
    magnitudes = [0, 10, 20, 40, 60, 80, 85, 88, 90, 100, 200, 500]
    naive_err, stable_err = [], []
    for m in magnitudes:
        x32 = np.array([[m, m - 1.0, m - 2.0]], dtype=np.float32)
        exact = log_softmax(np.array([[0.0, -1.0, -2.0]], dtype=np.float64))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            n = naive_log_softmax(x32).astype(np.float64)
        s = log_softmax(x32.astype(np.float64))
        naive_err.append(float(np.nan_to_num(np.abs(n - exact), nan=1e9, posinf=1e9).max()))
        stable_err.append(float(np.abs(s - exact).max()))
    return {"logit_magnitude": magnitudes, "naive_error": naive_err, "stable_error": stable_err}


def gradient_checks() -> dict[str, float]:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 12))
    g = rng.standard_normal((4, 12))
    gamma, beta = rng.standard_normal(12), rng.standard_normal(12)

    _, cache = layer_norm(x, gamma, beta)
    dx, dgamma, dbeta = layer_norm_backward(g, cache)
    ln_loss = lambda: float((layer_norm(x, gamma, beta)[0] * g).sum())  # noqa: E731
    out = {
        "layer_norm_dx": max_abs_error(dx, numeric_grad(ln_loss, x)),
        "layer_norm_dgamma": max_abs_error(dgamma, numeric_grad(ln_loss, gamma)),
        "layer_norm_dbeta": max_abs_error(dbeta, numeric_grad(ln_loss, beta)),
    }

    _, cache = rms_norm(x, gamma)
    dx, dgamma = rms_norm_backward(g, cache)
    rms_loss = lambda: float((rms_norm(x, gamma)[0] * g).sum())  # noqa: E731
    out["rms_norm_dx"] = max_abs_error(dx, numeric_grad(rms_loss, x))
    out["rms_norm_dgamma"] = max_abs_error(dgamma, numeric_grad(rms_loss, gamma))

    probs = softmax(x)
    w = rng.standard_normal((4, 12))
    out["softmax_vjp"] = max_abs_error(
        softmax_backward(w, probs), numeric_grad(lambda: float((softmax(x) * w).sum()), x)
    )
    return out


def norm_cost() -> dict[str, float]:
    """RMSNorm vs LayerNorm forward cost on a realistic activation tensor."""
    x = np.random.default_rng(0).standard_normal((32, 512, 1024))
    gamma = np.ones(1024)
    timings = {}
    for name, fn in (("layer_norm", lambda: layer_norm(x, gamma, gamma)),
                     ("rms_norm", lambda: rms_norm(x, gamma))):
        fn()  # warm up page faults / caches
        start = time.perf_counter()
        for _ in range(3):
            fn()
        timings[name] = round((time.perf_counter() - start) / 3 * 1000, 2)
    timings["rms_speedup"] = round(timings["layer_norm"] / timings["rms_norm"], 3)
    return timings


def attention_scale_effect() -> dict[str, float]:
    rng = np.random.default_rng(9)
    d_head, seq = 64, 32
    q, k, v = (rng.standard_normal((1, 1, seq, d_head)) for _ in range(3))
    _, scaled = scaled_dot_product_attention(q, k, v)
    unscaled = softmax(q @ np.swapaxes(k, -1, -2), axis=-1)
    ent = lambda w: float(-(w * np.log(w + 1e-12)).sum(axis=-1).mean())  # noqa: E731
    return {
        "d_head": d_head,
        "seq_len": seq,
        "uniform_entropy_nats": round(float(np.log(seq)), 3),
        "entropy_with_scale": round(ent(scaled), 3),
        "entropy_without_scale": round(ent(unscaled), 3),
        "max_weight_without_scale": round(float(unscaled.max()), 5),
    }


def plot(curve: dict[str, list[float]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=140)
    ax.plot(curve["logit_magnitude"], np.maximum(curve["naive_error"], 1e-18), "o-",
            lw=1.6, label="naive log(softmax(x)), float32")
    ax.plot(curve["logit_magnitude"], np.maximum(curve["stable_error"], 1e-18), "s-",
            lw=1.6, label="stable log_softmax")
    ax.axvline(88.7, ls="--", c="grey", lw=1)
    ax.text(90, 1e-6, "float32 exp overflow\n(x = 88.7)", fontsize=7, color="grey")
    ax.set(xlabel="logit magnitude", ylabel="max abs error vs exact",
           title="Naive vs stable log-softmax")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "stability.png")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-6, 6, 4001)
    payload = {
        "dtype_limits": dtype_table(),
        "log_softmax_error_curve": error_curve(),
        "gradient_check_max_abs_error": gradient_checks(),
        "norm_forward_ms_on_32x512x1024": norm_cost(),
        "attention_scale_effect": attention_scale_effect(),
        "gelu_tanh_approximation_max_abs_error": float(
            np.abs(gelu(x) - gelu(x, approximate=True)).max()
        ),
    }
    plot(payload["log_softmax_error_curve"])
    (RESULTS / "stability_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
