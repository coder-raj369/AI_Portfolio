"""Tests for evalharness modules."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from evalharness.llm_judge import DummyJudge, RUBRIC_CODE_QUALITY, RUBRIC_HELPFULNESS
from evalharness.pass_at_k import pass_at_k, pass_at_k_from_bools
from evalharness.registry import REGISTRY, TaskResult
from evalharness.regression import (
    DEFAULT_THRESHOLDS,
    Threshold,
    compare,
    load_baseline,
    save_baseline,
)
from evalharness.verifiers import (
    CodeVerifier,
    exact_match,
    expression_match,
    numeric_match,
)


# ---------------------------------------------------------------------------
# pass@k
# ---------------------------------------------------------------------------

class TestPassAtK:
    def test_all_correct(self):
        assert pass_at_k(100, 100, 1) == 1.0
        assert pass_at_k(100, 100, 50) == 1.0

    def test_all_incorrect(self):
        assert pass_at_k(100, 0, 1) == 0.0
        assert pass_at_k(100, 0, 50) == 0.0

    def test_half_correct_k1(self):
        # n=100, c=50, k=1 -> 50/100 = 0.5
        assert pass_at_k(100, 50, 1) == pytest.approx(0.5, abs=0.01)

    def test_known_value(self):
        # n=200, c=3, k=1 -> 3/200 = 0.015 exactly
        assert pass_at_k(200, 3, 1) == pytest.approx(0.015, abs=1e-6)

    def test_known_value_k100(self):
        # n=200, c=3, k=100 -> 1 - C(197,100)/C(200,100)
        # = 1 - (100*99*98)/(200*199*198) ≈ 0.877
        assert pass_at_k(200, 3, 100) == pytest.approx(0.8769, abs=0.001)
        # From Chen et al. paper: n=200, c=3, k=1 -> ~0.0149
        assert pass_at_k(200, 3, 1) == pytest.approx(0.014925, abs=1e-6)

    def test_known_value_k100(self):
        # n=200, c=3, k=100 -> 1 - C(197,100) / C(200,100) ~= 0.8769
        assert pass_at_k(200, 3, 100) == pytest.approx(0.8768844221105501, abs=1e-6)

    def test_from_bools(self):
        correct = [True, False, True, False, False]
        assert pass_at_k_from_bools(correct, 1) == pass_at_k(5, 2, 1)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            pass_at_k(-1, 0, 1)
        with pytest.raises(ValueError):
            pass_at_k(10, 11, 1)
        with pytest.raises(ValueError):
            pass_at_k(10, 5, 11)

    def test_underflow_protection(self):
        # Very small ratio should not underflow to 0
        assert pass_at_k(10_000, 1, 1) > 0
        assert pass_at_k(10_000, 1, 1) < 0.001


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_simple(self):
        assert exact_match("42", "42")
        assert not exact_match("42", "43")

    def test_strip(self):
        assert exact_match(" 42 ", "42", strip=True)
        assert not exact_match(" 42 ", "42", strip=False)


class TestNumericMatch:
    def test_exact(self):
        assert numeric_match("42", "42")

    def test_tolerance(self):
        assert numeric_match("3.14159", "3.14158")

    def test_extraction(self):
        assert numeric_match("The answer is 42.", "42")
        assert numeric_match("42", "The answer is 42.")

    def test_scientific(self):
        assert numeric_match("1.23e4", "12300")

    def test_no_number(self):
        assert not numeric_match("none", "42")

    def test_negative(self):
        assert numeric_match("-5", "-5.0")


class TestExpressionMatch:
    def test_addition(self):
        assert expression_match("2 + 2", "4")

    def test_multiplication(self):
        assert expression_match("3 * 4", "12")

    def test_power(self):
        assert expression_match("2 ** 3", "8")

    def test_inequivalent(self):
        assert not expression_match("2 + 2", "5")

    def test_unsafe_rejected(self):
        # __import__ should not work
        assert not expression_match("__import__('os')", "0")


class TestCodeVerifier:
    def test_correct_function(self):
        v = CodeVerifier(timeout_sec=1.0)
        code = "def solution(x):\n    return x * 2"

        def test_fn(fn):
            return fn(5) == 10

        assert v.verify(code, test_fn, entry_point="solution")

    def test_syntax_error(self):
        v = CodeVerifier(timeout_sec=1.0)
        code = "def solution(x)\n    return x"

        def test_fn(fn):
            return fn(1) == 1

        assert not v.verify(code, test_fn, entry_point="solution")

    def test_runtime_error(self):
        v = CodeVerifier(timeout_sec=1.0)
        code = "def solution(x):\n    return 1 / 0"

        def test_fn(fn):
            return fn(1) == 1

        assert not v.verify(code, test_fn, entry_point="solution")

    def test_timeout(self):
        v = CodeVerifier(timeout_sec=0.1)
        code = "def solution(x):\n    while True:\n        pass"

        def test_fn(fn):
            return fn(1) == 1

        assert not v.verify(code, test_fn, entry_point="solution")

    def test_missing_entry_point(self):
        v = CodeVerifier(timeout_sec=1.0)
        code = "x = 42"

        def test_fn(fn):
            return fn == 42

        # entry_point "solution" not defined -> returns None -> test_fn fails
        assert not v.verify(code, test_fn, entry_point="solution")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_register_and_run(self):
        # Use a fresh registry to avoid polluting the global one
        from evalharness.registry import Registry

        reg = Registry()

        @reg.register(
            name="dummy_math",
            description="Dummy math task",
            metric="exact_match",
        )
        def dummy_task(model) -> TaskResult:
            return TaskResult(
                task_name="dummy_math",
                metric="exact_match",
                score=0.75,
            )

        spec, fn = reg.get("dummy_math")
        assert spec.name == "dummy_math"
        assert spec.metric == "exact_match"

        result = reg.run("dummy_math", None)
        assert result.score == 0.75

    def test_duplicate_registration_fails(self):
        from evalharness.registry import Registry

        reg = Registry()

        @reg.register(name="dup", description="", metric="")
        def task1(_):
            return TaskResult("dup", "", 0.0)

        with pytest.raises(ValueError):

            @reg.register(name="dup", description="", metric="")
            def task2(_):
                return TaskResult("dup", "", 0.0)

    def test_list_tasks_by_tag(self):
        from evalharness.registry import Registry

        reg = Registry()

        @reg.register(name="math", description="", metric="", tags=("math",))
        def math_task(_):
            return TaskResult("math", "", 0.0)

        @reg.register(name="code", description="", metric="", tags=("code",))
        def code_task(_):
            return TaskResult("code", "", 0.0)

        assert len(reg.list_tasks(tag="math")) == 1
        assert len(reg.list_tasks(tag="code")) == 1
        assert len(reg.list_tasks()) == 2


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class TestDummyJudge:
    def test_determinism(self):
        judge = DummyJudge(seed=0.5)
        s1 = judge.evaluate(RUBRIC_HELPFULNESS, "Q", "A1")
        s2 = judge.evaluate(RUBRIC_HELPFULNESS, "Q", "A1")
        assert s1.scores == s2.scores

    def test_different_answers_different_scores(self):
        judge = DummyJudge(seed=0.5)
        s1 = judge.evaluate(RUBRIC_HELPFULNESS, "Q", "A1")
        s2 = judge.evaluate(RUBRIC_HELPFULNESS, "Q", "A2")
        # Extremely unlikely to be identical with SHA256 hash
        assert s1.scores != s2.scores

    def test_score_range(self):
        judge = DummyJudge(seed=0.5)
        s = judge.evaluate(RUBRIC_HELPFULNESS, "Q", "A")
        for item in RUBRIC_HELPFULNESS.items:
            assert 0 <= s.scores[item.name] <= item.max_score

    def test_total(self):
        judge = DummyJudge(seed=0.5)
        s = judge.evaluate(RUBRIC_CODE_QUALITY, "Q", "A")
        assert s.total() == sum(s.scores.values())

    def test_to_dict(self):
        judge = DummyJudge(seed=0.5)
        s = judge.evaluate(RUBRIC_HELPFULNESS, "Q", "A")
        d = s.to_dict()
        assert d["rubric_name"] == "helpfulness"
        assert "total" in d


class TestLLMJudgePrompt:
    def test_build_prompt_includes_rubric(self):
        from evalharness.llm_judge import LLMJudge

        judge = LLMJudge()
        prompt = judge._build_prompt(RUBRIC_HELPFULNESS, "Q1", "A1")
        assert "relevance" in prompt
        assert "accuracy" in prompt
        assert "Q1" in prompt
        assert "A1" in prompt

    def test_parse_response_json(self):
        from evalharness.llm_judge import LLMJudge

        judge = LLMJudge()
        raw = '{"scores": {"relevance": 0.8, "accuracy": 0.9}, "explanation": "good"}'
        scores = judge._parse_response(raw, RUBRIC_HELPFULNESS)
        assert scores["relevance"] == 0.8
        assert scores["accuracy"] == 0.9

    def test_parse_response_fallback(self):
        from evalharness.llm_judge import LLMJudge

        judge = LLMJudge()
        raw = "relevance: 0.7, accuracy: 0.8"
        scores = judge._parse_response(raw, RUBRIC_HELPFULNESS)
        assert scores["relevance"] == 0.7
        assert scores["accuracy"] == 0.8


# ---------------------------------------------------------------------------
# Regression gating
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_no_regression_when_improved(self):
        t = Threshold(absolute=0.05, relative=0.10, higher_is_better=True)
        assert not t.is_regression(0.5, 0.6)

    def test_regression_on_absolute(self):
        t = Threshold(absolute=0.05, relative=0.10, higher_is_better=True)
        assert t.is_regression(0.5, 0.44)

    def test_no_regression_within_tol(self):
        t = Threshold(absolute=0.05, relative=0.10, higher_is_better=True)
        assert not t.is_regression(0.5, 0.48)

    def test_lower_is_better(self):
        t = Threshold(absolute=0.05, relative=0.10, higher_is_better=False)
        assert t.is_regression(0.5, 0.6)  # higher is worse
        assert not t.is_regression(0.5, 0.4)  # lower is better


class TestCompare:
    def test_no_regression(self):
        baseline = {"task": TaskResult("task", "exact_match", 0.80)}
        current = {"task": TaskResult("task", "exact_match", 0.82)}
        report = compare(current, baseline)
        assert not report.any_regression()

    def test_regression_detected(self):
        baseline = {"task": TaskResult("task", "exact_match", 0.80)}
        current = {"task": TaskResult("task", "exact_match", 0.70)}
        report = compare(current, baseline)
        assert report.any_regression()
        assert report.regressed_tasks() == ["task"]

    def test_new_task_no_regression(self):
        baseline = {"old": TaskResult("old", "exact_match", 0.80)}
        current = {
            "old": TaskResult("old", "exact_match", 0.80),
            "new": TaskResult("new", "exact_match", 0.50),
        }
        report = compare(current, baseline)
        assert not report.any_regression()

    def test_round_trip_json(self):
        baseline = {"task": TaskResult("task", "exact_match", 0.80)}
        current = {"task": TaskResult("task", "exact_match", 0.75)}
        report = compare(current, baseline)
        d = report.to_dict()
        assert d["regressed"] is True
        assert len(d["comparisons"]) == 1
        assert d["comparisons"][0]["delta"] == pytest.approx(-0.05, abs=1e-12)


class TestBaselineIO:
    def test_save_load_roundtrip(self):
        results = {
            "math": TaskResult("math", "exact_match", 0.75, {"n": 100}),
            "code": TaskResult("code", "pass@1", 0.42),
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_baseline(results, path)
            loaded = load_baseline(path)
            assert loaded["math"].score == 0.75
            assert loaded["math"].details == {"n": 100}
            assert loaded["code"].score == 0.42
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Integration / smoke
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline(self):
        """Simulate: run a task, judge it, and check for regression."""
        from evalharness.registry import Registry

        reg = Registry()

        @reg.register(
            name="toy_math",
            description="Toy math task",
            metric="exact_match",
        )
        def toy_task(model) -> TaskResult:
            # Simulate model answering 3/4 correctly
            return TaskResult(
                task_name="toy_math",
                metric="exact_match",
                score=0.75,
                details={"n": 4, "correct": 3},
            )

        result = reg.run("toy_math", None)
        assert result.score == 0.75

        # No regression against itself
        baseline = {"toy_math": result}
        current = {"toy_math": result}
        report = compare(current, baseline)
        assert not report.any_regression()
