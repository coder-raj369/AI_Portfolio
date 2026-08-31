"""Validate all evalharness modules end-to-end.

Runs the test suite and produces a results ledger for the README.
No external API keys or models required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from evalharness.llm_judge import DummyJudge, RUBRIC_HELPFULNESS
from evalharness.pass_at_k import pass_at_k
from evalharness.verifiers import exact_match, numeric_match


def main() -> int:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("evalharness validation")
    print("=" * 60)

    # 1. pass@k sanity checks
    print("\n[pass@k sanity checks]")
    passk_results = {
        "pass_at_k(200, 3, 1)": pass_at_k(200, 3, 1),
        "pass_at_k(200, 3, 100)": pass_at_k(200, 3, 100),
        "pass_at_k(100, 100, 1)": pass_at_k(100, 100, 1),
        "pass_at_k(100, 0, 1)": pass_at_k(100, 0, 1),
    }
    for k, v in passk_results.items():
        print(f"  {k} = {v:.6f}")

    # 2. Verifier accuracy
    print("\n[verifier checks]")
    verifier_results = {
        "exact_match_42_42": exact_match("42", "42"),
        "exact_match_strip": exact_match(" 42 ", "42"),
        "numeric_3_14159": numeric_match("3.14159", "3.14158"),
        "numeric_extract": numeric_match("The answer is 42.", "42"),
        "numeric_no_number": numeric_match("none", "42"),
    }
    for k, v in verifier_results.items():
        print(f"  {k}: {v}")

    # 3. Dummy judge determinism
    print("\n[dummy judge determinism]")
    judge = DummyJudge(seed=0.5)
    s1 = judge.evaluate(RUBRIC_HELPFULNESS, "What is 2+2?", "4")
    s2 = judge.evaluate(RUBRIC_HELPFULNESS, "What is 2+2?", "4")
    print(f"  Same answer twice: scores equal = {s1.scores == s2.scores}")
    print(f"  Score total: {s1.total():.2f} / 3.0")

    # 4. Regression gating simulation
    print("\n[regression gating]")
    from evalharness.registry import TaskResult
    from evalharness.regression import compare

    baseline = {"math": TaskResult("math", "exact_match", 0.80)}
    current_ok = {"math": TaskResult("math", "exact_match", 0.79)}
    current_bad = {"math": TaskResult("math", "exact_match", 0.70)}

    report_ok = compare(current_ok, baseline)
    report_bad = compare(current_bad, baseline)
    print(f"  0.80 -> 0.79: regressed = {report_ok.any_regression()}")
    print(f"  0.80 -> 0.70: regressed = {report_bad.any_regression()}")

    # Write results
    output = {
        "pass_at_k": passk_results,
        "verifiers": {k: bool(v) for k, v in verifier_results.items()},
        "judge": {
            "deterministic": s1.scores == s2.scores,
            "total": s1.total(),
        },
        "regression": {
            "small_drop_ok": not report_ok.any_regression(),
            "large_drop_flagged": report_bad.any_regression(),
        },
    }

    out_path = results_dir / "validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote results to {out_path}")

    # Run pytest for the module
    print("\n[running pytest]")
    test_dir = Path(__file__).parent.parent / "tests"
    rc = subprocess.call([sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
