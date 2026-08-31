"""Task registry — decorator-based registration of evaluation tasks."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable


@dataclasses.dataclass(frozen=True, slots=True)
class TaskSpec:
    """Static metadata for an evaluation task."""

    name: str
    description: str
    metric: str  # primary metric name, e.g. "pass@1", "exact_match"
    higher_is_better: bool = True
    tags: tuple[str, ...] = ()


@dataclasses.dataclass(slots=True)
class TaskResult:
    """Result of running a single task."""

    task_name: str
    metric: str
    score: float
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "metric": self.metric,
            "score": self.score,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskResult:
        return cls(
            task_name=d["task_name"],
            metric=d["metric"],
            score=d["score"],
            details=d.get("details", {}),
        )


class Registry:
    """Global task registry.

    Tasks register themselves via the `@register` decorator:

        @register(
            name="grade_school_math",
            description="GSM8K-style grade-school math",
            metric="exact_match",
        )
        def eval_gsm8k(model) -> TaskResult:
            ...
    """

    def __init__(self) -> None:
        self._tasks: dict[str, tuple[TaskSpec, Callable[..., TaskResult]]] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        metric: str,
        higher_is_better: bool = True,
        tags: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., TaskResult]], Callable[..., TaskResult]]:
        """Decorator that registers an evaluation function."""
        spec = TaskSpec(
            name=name,
            description=description,
            metric=metric,
            higher_is_better=higher_is_better,
            tags=tags,
        )

        def decorator(fn: Callable[..., TaskResult]) -> Callable[..., TaskResult]:
            if name in self._tasks:
                raise ValueError(f"Task '{name}' already registered")
            self._tasks[name] = (spec, fn)
            return fn

        return decorator

    def get(self, name: str) -> tuple[TaskSpec, Callable[..., TaskResult]]:
        if name not in self._tasks:
            raise KeyError(f"No task named '{name}'")
        return self._tasks[name]

    def list_tasks(self, tag: str | None = None) -> list[TaskSpec]:
        specs = [spec for spec, _ in self._tasks.values()]
        if tag is not None:
            specs = [s for s in specs if tag in s.tags]
        return specs

    def run(
        self, name: str, *args: Any, **kwargs: Any
    ) -> TaskResult:
        spec, fn = self.get(name)
        result = fn(*args, **kwargs)
        if result.task_name != spec.name:
            raise ValueError(
                f"Task '{name}' returned result for '{result.task_name}'"
            )
        if result.metric != spec.metric:
            raise ValueError(
                f"Task '{name}' expects metric '{spec.metric}', "
                f"got '{result.metric}'"
            )
        return result

    def run_suite(
        self, names: list[str] | None = None, *args: Any, **kwargs: Any
    ) -> dict[str, TaskResult]:
        if names is None:
            names = list(self._tasks.keys())
        return {name: self.run(name, *args, **kwargs) for name in names}


# Global singleton registry
REGISTRY = Registry()
register = REGISTRY.register
