"""LLM-judge framework — rubric-based evaluation with an external model.

This module provides the *interface* and *aggregators* for LLM-as-judge
scoring. The actual model client is injected, so the harness stays
dependency-light and testable without API keys.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Protocol


class JudgeClient(Protocol):
    """Protocol for an LLM judge client."""

    def judge(self, prompt: str, **kwargs: Any) -> str:
        """Return the raw text response from the judge model."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class RubricItem:
    """A single criterion in a rubric."""

    name: str
    description: str
    max_score: float = 1.0


@dataclasses.dataclass(frozen=True, slots=True)
class Rubric:
    """A scoring rubric composed of criteria."""

    name: str
    items: tuple[RubricItem, ...]

    def max_total(self) -> float:
        return sum(item.max_score for item in self.items)


# Built-in rubrics
RUBRIC_HELPFULNESS = Rubric(
    name="helpfulness",
    items=(
        RubricItem("relevance", "Answer addresses the user's question", 1.0),
        RubricItem("accuracy", "Factual claims are correct", 1.0),
        RubricItem("completeness", "No important aspect is omitted", 1.0),
    ),
)

RUBRIC_CODE_QUALITY = Rubric(
    name="code_quality",
    items=(
        RubricItem("correctness", "Code solves the stated problem", 1.0),
        RubricItem("readability", "Code is clear and well-structured", 1.0),
        RubricItem("efficiency", "Algorithmic choices are reasonable", 1.0),
    ),
)


@dataclasses.dataclass(slots=True)
class JudgeScore:
    """Result of a single judge evaluation."""

    rubric_name: str
    scores: dict[str, float]  # item name -> score
    raw_response: str
    explanation: str = ""

    def total(self) -> float:
        return sum(self.scores.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_name": self.rubric_name,
            "scores": self.scores,
            "total": self.total(),
            "raw_response": self.raw_response,
            "explanation": self.explanation,
        }


class LLMJudge:
    """Rubric-based LLM judge.

    Usage:
        judge = LLMJudge(client=my_openai_client)
        score = judge.evaluate(
            rubric=RUBRIC_HELPFULNESS,
            question="What is the capital of France?",
            answer="Paris is the capital.",
        )
    """

    def __init__(self, client: JudgeClient | None = None) -> None:
        self.client = client

    def _build_prompt(
        self, rubric: Rubric, question: str, answer: str, reference: str = ""
    ) -> str:
        lines = [
            "You are an expert evaluator. Score the answer below using the provided rubric.",
            "",
            f"Rubric: {rubric.name}",
        ]
        for item in rubric.items:
            lines.append(f"  - {item.name} (0 to {item.max_score}): {item.description}")
        lines.extend([
            "",
            f"Question: {question}",
            f"Answer: {answer}",
        ])
        if reference:
            lines.append(f"Reference answer: {reference}")
        lines.extend([
            "",
            "Respond in JSON format:",
            json.dumps(
                {
                    "scores": {item.name: "<float>" for item in rubric.items},
                    "explanation": "<brief reasoning>",
                },
                indent=2,
            ),
        ])
        return "\n".join(lines)

    def _parse_response(self, raw: str, rubric: Rubric) -> dict[str, float]:
        """Parse JSON scores from judge response; fallback to 0 on failure."""
        # Try to extract JSON block
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        try:
            data = json.loads(raw.strip())
            scores = data.get("scores", {})
            return {
                item.name: float(scores.get(item.name, 0.0))
                for item in rubric.items
            }
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to find numbers in the text
            import re

            found: dict[str, float] = {}
            for item in rubric.items:
                pattern = rf"{re.escape(item.name)}\D*([0-9]+(?:\.[0-9]+)?)"
                m = re.search(pattern, raw, re.IGNORECASE)
                found[item.name] = float(m.group(1)) if m else 0.0
            return found

    def evaluate(
        self,
        rubric: Rubric,
        question: str,
        answer: str,
        reference: str = "",
        **client_kwargs: Any,
    ) -> JudgeScore:
        if self.client is None:
            raise RuntimeError(
                "LLMJudge requires a client. Pass one at init or use DummyJudge."
            )
        prompt = self._build_prompt(rubric, question, answer, reference)
        raw = self.client.judge(prompt, **client_kwargs)
        scores = self._parse_response(raw, rubric)
        # Try to extract explanation
        explanation = ""
        try:
            data = json.loads(raw.strip())
            explanation = data.get("explanation", "")
        except Exception:
            pass
        return JudgeScore(
            rubric_name=rubric.name,
            scores=scores,
            raw_response=raw,
            explanation=explanation,
        )


class DummyJudge:
    """Deterministic dummy judge for testing and CI.

    Returns fixed scores based on a hash of the answer, so results are
    reproducible without an API call.
    """

    def __init__(self, seed: float = 0.5) -> None:
        self.seed = seed

    def evaluate(
        self,
        rubric: Rubric,
        question: str,
        answer: str,
        reference: str = "",
        **_: Any,
    ) -> JudgeScore:
        import hashlib

        h = hashlib.sha256(answer.encode()).hexdigest()
        scores = {}
        for i, item in enumerate(rubric.items):
            # Deterministic pseudo-random score in [0, max_score]
            val = int(h[i * 4 : i * 4 + 4], 16) / 65535.0
            scores[item.name] = round(val * item.max_score, 2)
        return JudgeScore(
            rubric_name=rubric.name,
            scores=scores,
            raw_response="<dummy judge — deterministic hash-based scores>",
            explanation="Generated by DummyJudge for testing.",
        )
