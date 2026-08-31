"""evalharness — lightweight LLM evaluation harness.

Provides:
- Task registry with decorator-based registration
- pass@k for code generation
- Verifier scoring for checkable answers (math, code)
- LLM-judge framework with rubric-based scoring
- Regression gating against baselines

All metrics are deterministic and reproducible; no API keys needed for the
built-in verifiers. LLM-judge requires an external model client (stub provided).
"""

__version__ = "0.1.0"
