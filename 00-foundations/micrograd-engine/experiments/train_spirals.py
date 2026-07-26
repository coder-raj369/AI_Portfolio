"""Train an MLP on interleaved spirals using nothing but nanograd.

Writes `experiments/results/metrics.json`, a loss curve and a decision-boundary
plot. Every number quoted in this module's README comes from this script.

Run: `python experiments/train_spirals.py`
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

RESULTS = Path(__file__).parent / "results"
STEPS = 1500
BATCH = 64
LR = 0.1
MOMENTUM = 0.9


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    return float((model(Tensor(X)).data.argmax(axis=1) == y).mean())


def train() -> dict[str, object]:
    X, y = make_spirals(n_per_class=180, seed=1)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=1)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    model = MLP([2, 32, 32, 3], activation="tanh", seed=1)
    params = model.parameters()
    velocity = [np.zeros_like(p.data) for p in params]
    rng = np.random.default_rng(0)

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for step in range(STEPS):
        idx = rng.choice(Xtr.shape[0], size=BATCH, replace=False)
        loss = cross_entropy(model(Tensor(Xtr[idx])), ytr[idx])
        model.zero_grad()
        loss.backward()
        for p, v in zip(params, velocity):
            v *= MOMENTUM
            v -= LR * p.grad
            p.data += v
        if step % 25 == 0 or step == STEPS - 1:
            history.append({"step": step, "batch_loss": loss.item()})
    wall = time.perf_counter() - start

    metrics = {
        "engine": "nanograd (NumPy autodiff, no torch)",
        "task": "3-class interleaved spirals, noise=0.08, 540 points",
        "architecture": "MLP 2-32-32-3, tanh",
        "parameters": model.num_parameters(),
        "optimiser": f"SGD momentum={MOMENTUM}, lr={LR}, batch={BATCH}, steps={STEPS}",
        "initial_batch_loss": history[0]["batch_loss"],
        "final_batch_loss": history[-1]["batch_loss"],
        "chance_loss": float(np.log(3)),
        "train_accuracy": accuracy(model, Xtr, ytr),
        "test_accuracy": accuracy(model, Xte, yte),
        "train_seconds_cpu": round(wall, 2),
        "steps_per_second": round(STEPS / wall, 1),
        "history": history,
    }
    metrics["baseline_sklearn"] = sklearn_baseline(Xtr, ytr, Xte, yte)
    plot_curves(history, model, Xtr, ytr)
    return metrics


def sklearn_baseline(Xtr, ytr, Xte, yte) -> dict[str, float]:
    """A deliberately *strong* reference point: sklearn's MLP with L-BFGS.

    Beating a crippled baseline proves nothing, so this uses the solver and
    settings that perform best on this task rather than library defaults
    (whose early-stopping heuristics stall at ~0.42 train accuracy here).
    """
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(
        hidden_layer_sizes=(32, 32), activation="tanh", solver="lbfgs",
        max_iter=3000, random_state=1,
    ).fit(Xtr, ytr)
    return {
        "impl": "sklearn MLPClassifier (32,32) tanh, lbfgs",
        "train_accuracy": float(clf.score(Xtr, ytr)),
        "test_accuracy": float(clf.score(Xte, yte)),
    }


def plot_curves(history, model: MLP, X: np.ndarray, y: np.ndarray) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    steps = [h["step"] for h in history]
    losses = [h["batch_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=140)
    ax.plot(steps, losses, lw=1.6)
    ax.axhline(np.log(3), ls="--", c="grey", lw=1, label="chance = ln(3)")
    ax.set(xlabel="step", ylabel="minibatch cross-entropy", title="nanograd: spirals training")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curve.png")
    plt.close(fig)

    pad = 0.5
    gx, gy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300),
    )
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    pred = model(Tensor(grid)).data.argmax(axis=1).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=140)
    ax.contourf(gx, gy, pred, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.25, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], c=y, s=7, cmap="viridis", edgecolors="none")
    ax.set(title="learned decision boundary", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(RESULTS / "decision_boundary.png")
    plt.close(fig)


def main() -> None:
    metrics = train()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
