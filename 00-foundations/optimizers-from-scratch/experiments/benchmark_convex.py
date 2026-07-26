"""Compare optimisers on problems where the answer is known.

Two objectives:
  1. An ill-conditioned quadratic (condition number 100) - the classic case that
     separates adaptive methods from plain SGD.
  2. Rosenbrock - non-convex, curved valley, punishes badly scaled steps.

Each optimiser gets its own learning-rate sweep, because comparing optimisers at
one shared learning rate measures the learning rate, not the optimiser.

Run: `python experiments/benchmark_convex.py`
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nanoptim import SGD, Adam, AdamW, ConstantLR, CosineWithWarmup  # noqa: E402
from nanoptim import Lion, Parameter, WarmupStableDecay  # noqa: E402

RESULTS = Path(__file__).parent / "results"
STEPS = 500
LR_GRID = [10.0**e for e in range(-4, 1)] + [3 * 10.0**e for e in range(-4, 0)]

BUILDERS: dict[str, Callable[[list[Parameter], float], object]] = {
    "SGD": lambda p, lr: SGD(p, lr=lr),
    "SGD+momentum": lambda p, lr: SGD(p, lr=lr, momentum=0.9),
    "SGD+Nesterov": lambda p, lr: SGD(p, lr=lr, momentum=0.9, nesterov=True),
    "Adam": lambda p, lr: Adam(p, lr=lr),
    "AdamW(wd=0.01)": lambda p, lr: AdamW(p, lr=lr, weight_decay=0.01),
    "Lion": lambda p, lr: Lion(p, lr=lr),
}


def quadratic(condition: float = 100.0, dim: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    A_eig = np.logspace(0, np.log10(condition), dim)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    A = q @ np.diag(A_eig) @ q.T
    x0 = rng.standard_normal(dim) * 2.0

    def f(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(0.5 * x @ A @ x), A @ x

    return f, x0, "quadratic (cond=100, d=20)"


def rosenbrock():
    x0 = np.array([-1.2, 1.0])

    def f(x: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = x[0], x[1]
        loss = (1 - a) ** 2 + 100.0 * (b - a * a) ** 2
        grad = np.array(
            [-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)]
        )
        return float(loss), grad

    return f, x0, "Rosenbrock (2-d, non-convex)"


def run(objective, builder, lr: float, steps: int = STEPS, schedule=None) -> list[float]:
    f, x0, _ = objective()
    p = Parameter(x0)
    opt = builder([p], lr)
    losses = []
    for step in range(steps):
        if schedule is not None:
            opt.set_lr(schedule(step))
        loss, grad = f(p.data)
        if not np.isfinite(loss) or loss > 1e12:
            return losses + [float("inf")] * (steps - len(losses))
        p.grad = grad
        opt.step()
        losses.append(loss)
    return losses


def sweep(objective) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, builder in BUILDERS.items():
        best = None
        for lr in sorted(LR_GRID):
            losses = run(objective, builder, lr)
            final = losses[-1]
            if best is None or (np.isfinite(final) and final < best["final_loss"]):
                target = 1e-6 * losses[0]
                hit = next((i for i, v in enumerate(losses) if v < target), None)
                best = {
                    "best_lr": lr,
                    "final_loss": final,
                    "initial_loss": losses[0],
                    "steps_to_1e-6_of_initial": hit,
                    "curve": losses,
                }
        out[name] = best
    return out


def plot(results: dict[str, dict], title: str, filename: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=140)
    for name, r in results.items():
        ax.plot(np.maximum(r["curve"], 1e-16), lw=1.5, label=f"{name} (lr={r['best_lr']:g})")
    ax.set(xlabel="step", ylabel="loss", title=title)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(RESULTS / filename)
    plt.close(fig)


def schedule_comparison() -> dict[str, dict]:
    """Same optimiser, three schedules: isolates the schedule's contribution."""
    peak = 0.3
    schedules = {
        "constant": ConstantLR(peak),
        "cosine+warmup(10%)": CosineWithWarmup(peak, STEPS, warmup_steps=STEPS // 10),
        "WSD(2%/10%)": WarmupStableDecay(peak, STEPS),
    }
    out = {}
    for name, sched in schedules.items():
        curve = run(quadratic, BUILDERS["Adam"], peak, schedule=sched)
        out[name] = {"final_loss": curve[-1], "curve": curve, "best_lr": peak}
    plot(out, "Adam on the quadratic: schedule matters", "schedule_comparison.png")
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    quad = sweep(quadratic)
    rosen = sweep(rosenbrock)
    plot(quad, "Optimisers on an ill-conditioned quadratic", "convex_quadratic.png")
    plot(rosen, "Optimisers on Rosenbrock", "rosenbrock.png")
    sched = schedule_comparison()

    payload = {
        "steps": STEPS,
        "lr_grid": sorted(LR_GRID),
        "quadratic": {
            k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in quad.items()
        },
        "rosenbrock": {
            k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in rosen.items()
        },
        "schedules_on_quadratic": {
            k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in sched.items()
        },
    }
    (RESULTS / "convex_benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
