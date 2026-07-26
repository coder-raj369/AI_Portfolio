"""Same optimisers, now driving a real network through the sibling autodiff engine.

Convex benchmarks can flatter adaptive methods; this checks whether the ranking
survives on a small non-convex classification problem. Each optimiser is tuned
over its own learning-rate grid and every run uses identical data, init seed and
batch order, so the only variable is the update rule.

Requires the sibling module `00-foundations/micrograd-engine` on the path
(handled by the root pyproject / this project's `[tool.uv.sources]`).

Run: `python experiments/benchmark_spirals.py`
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from nanograd import MLP, Tensor, cross_entropy, make_spirals, train_test_split  # noqa: E402

from nanoptim import SGD, Adam, AdamW, CosineWithWarmup, Lion  # noqa: E402

RESULTS = Path(__file__).parent / "results"
STEPS = 1000
BATCH = 64

BUILDERS = {
    "SGD": lambda p, lr: SGD(p, lr=lr),
    "SGD+momentum": lambda p, lr: SGD(p, lr=lr, momentum=0.9),
    "SGD+Nesterov": lambda p, lr: SGD(p, lr=lr, momentum=0.9, nesterov=True),
    "Adam": lambda p, lr: Adam(p, lr=lr),
    "AdamW(wd=0.01)": lambda p, lr: AdamW(p, lr=lr, weight_decay=0.01),
    "Lion": lambda p, lr: Lion(p, lr=lr),
}
LR_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def data():
    X, y = make_spirals(n_per_class=180, seed=1)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=1)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    return (Xtr - mu) / sd, ytr, (Xte - mu) / sd, yte


def train_once(builder, lr: float, schedule=None) -> dict:
    Xtr, ytr, Xte, yte = data()
    model = MLP([2, 32, 32, 3], activation="tanh", seed=1)
    opt = builder(model.parameters(), lr)
    rng = np.random.default_rng(0)
    curve, steps_to_95 = [], None
    start = time.perf_counter()
    for step in range(STEPS):
        if schedule is not None:
            opt.set_lr(schedule(step))
        idx = rng.choice(Xtr.shape[0], size=BATCH, replace=False)
        loss = cross_entropy(model(Tensor(Xtr[idx])), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(loss.item())
        if steps_to_95 is None and step % 20 == 0:
            acc = float((model(Tensor(Xtr)).data.argmax(axis=1) == ytr).mean())
            if acc >= 0.95:
                steps_to_95 = step
    wall = time.perf_counter() - start
    return {
        "lr": lr,
        "final_batch_loss": curve[-1],
        "train_accuracy": float((model(Tensor(Xtr)).data.argmax(axis=1) == ytr).mean()),
        "test_accuracy": float((model(Tensor(Xte)).data.argmax(axis=1) == yte).mean()),
        "steps_to_95pct_train": steps_to_95,
        "seconds": round(wall, 2),
        "curve": curve,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for name, builder in BUILDERS.items():
        best = None
        for lr in LR_GRID:
            r = train_once(builder, lr)
            score = (r["test_accuracy"], -r["final_batch_loss"])
            if best is None or score > (best["test_accuracy"], -best["final_batch_loss"]):
                best = r
        results[name] = best
        print(f"{name:16s} lr={best['lr']:<6g} test={best['test_accuracy']:.3f} "
              f"steps@95%={best['steps_to_95pct_train']} loss={best['final_batch_loss']:.4f}")

    # Schedule ablation on the strongest optimiser configuration.
    best_name = max(results, key=lambda k: results[k]["test_accuracy"])
    sched = CosineWithWarmup(
        results[best_name]["lr"], total_steps=STEPS, warmup_steps=STEPS // 20
    )
    with_sched = train_once(BUILDERS[best_name], results[best_name]["lr"], schedule=sched)
    print(f"{best_name} + cosine warmup: test={with_sched['test_accuracy']:.3f} "
          f"loss={with_sched['final_batch_loss']:.4f}")

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=140)
    for name, r in results.items():
        smooth = np.convolve(r["curve"], np.ones(25) / 25, mode="valid")
        ax.plot(smooth, lw=1.4, label=f"{name} (lr={r['lr']:g})")
    ax.set(xlabel="step", ylabel="minibatch cross-entropy (25-step mean)",
           title="Optimisers on spirals via nanograd")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "spirals_optimisers.png")
    plt.close(fig)

    payload = {
        "setup": f"MLP 2-32-32-3 tanh (1251 params), batch={BATCH}, steps={STEPS}, "
                 "identical seed/batch order across optimisers",
        "lr_grid": LR_GRID,
        "per_optimiser_best": {
            k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in results.items()
        },
        "schedule_ablation": {
            "optimiser": best_name,
            "constant_lr": {
                k: v for k, v in results[best_name].items() if k != "curve"
            },
            "cosine_warmup": {k: v for k, v in with_sched.items() if k != "curve"},
        },
    }
    (RESULTS / "spirals_benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
