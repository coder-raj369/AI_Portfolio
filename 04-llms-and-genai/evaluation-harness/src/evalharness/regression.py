"""Regression gating — compare current results against baselines and fail on regression.

Typical workflow:

    1. Run suite, save results to `baseline.json`
    2. After code changes, run suite again
    3. `compare(current, baseline)` returns a Report with pass/fail per task

A task fails if its score moves in the wrong direction by more than the
configured tolerance.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from evalharness.registry import TaskResult


@dataclasses.dataclass(frozen=True, slots=True)
class Threshold:
    """Regression threshold for a metric.

    If higher_is_better=True (default), a drop of more than `absolute` or
    more than `relative` fraction is a regression.
    """

    absolute: float = 0.0
    relative: float = 0.0
    higher_is_better: bool = True

    def is_regression(self, baseline: float, current: float) -> bool:
        delta = current - baseline
        if not self.higher_is_better:
            delta = -delta
        if delta < -self.absolute:
            return True
        if baseline != 0 and delta / abs(baseline) < -self.relative:
            return True
        return False


DEFAULT_THRESHOLDS: dict[str, Threshold] = {
    "pass@1": Threshold(absolute=0.02, relative=0.05, higher_is_better=True),
    "exact_match": Threshold(absolute=0.02, relative=0.05, higher_is_better=True),
    "llm_judge": Threshold(absolute=0.05, relative=0.10, higher_is_better=True),
}


@dataclasses.dataclass(slots=True)
class Comparison:
    """Result of comparing one task."""

    task_name: str
    baseline: float
    current: float
    delta: float
    regressed: bool
    threshold: Threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "baseline": self.baseline,
            "current": self.current,
            "delta": self.delta,
            "regressed": self.regressed,
        }


@dataclasses.dataclass(slots=True)
class Report:
    """Full regression report."""

    comparisons: list[Comparison]

    def any_regression(self) -> bool:
        return any(c.regressed for c in self.comparisons)

    def regressed_tasks(self) -> list[str]:
        return [c.task_name for c in self.comparisons if c.regressed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "regressed": self.any_regression(),
            "regressed_tasks": self.regressed_tasks(),
            "comparisons": [c.to_dict() for c in self.comparisons],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def compare(
    current: dict[str, TaskResult],
    baseline: dict[str, TaskResult],
    thresholds: dict[str, Threshold] | None = None,
) -> Report:
    """Compare current results against baseline.

    Args:
        current: mapping task_name -> TaskResult
        baseline: mapping task_name -> TaskResult
        thresholds: per-metric thresholds; falls back to DEFAULT_THRESHOLDS
            then a zero threshold.

    Returns:
        Report with regression flags.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    comps: list[Comparison] = []
    for name in sorted(set(current) | set(baseline)):
        cur = current.get(name)
        base = baseline.get(name)
        if cur is None:
            # Task removed — not a regression, but noted
            continue
        if base is None:
            # New task — no baseline to regress against
            continue

        thresh = thresholds.get(cur.metric, Threshold())
        regressed = thresh.is_regression(base.score, cur.score)
        comps.append(
            Comparison(
                task_name=name,
                baseline=base.score,
                current=cur.score,
                delta=cur.score - base.score,
                regressed=regressed,
                threshold=thresh,
            )
        )

    return Report(comparisons=comps)


def load_baseline(path: str) -> dict[str, TaskResult]:
    with open(path) as f:
        data = json.load(f)
    return {k: TaskResult.from_dict(v) for k, v in data.items()}


def save_baseline(results: dict[str, TaskResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
