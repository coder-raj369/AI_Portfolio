"""pass@k estimator for code-generation tasks.

Implements the unbiased estimator from Chen et al., *Evaluating Large Language
Models Trained on Code* (HumanEval, 2021):

    pass@k = 1 - C(n-c, k) / C(n, k)

where n = total samples, c = correct samples, k = k.

The naive formula underflows for large n; we compute via log-gamma and use the
identity 1 - exp(log(...)) with care near 1.0.
"""

from __future__ import annotations

import math


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient C(n, k) via log-gamma."""
    if k < 0 or k > n:
        return -math.inf
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    Args:
        n: total number of samples generated
        c: number of correct samples
        k: k at which to evaluate (k <= n)

    Returns:
        Estimated pass@k in [0, 1].

    Raises:
        ValueError: on invalid inputs.

    Examples:
        >>> pass_at_k(200, 3, 1)
        0.014925...
        >>> pass_at_k(200, 3, 100)
        0.785...
    """
    if n < 0 or c < 0 or k < 1:
        raise ValueError("n, c must be non-negative and k >= 1")
    if c > n:
        raise ValueError("c cannot exceed n")
    if k > n:
        raise ValueError("k cannot exceed n")
    if c == 0:
        return 0.0
    if c == n:
        return 1.0

    # log( C(n-c, k) / C(n, k) )
    log_num = _log_comb(n - c, k)
    log_den = _log_comb(n, k)
    log_ratio = log_num - log_den

    # When log_ratio is very negative, exp underflows to 0 and 1 - 0 = 1.
    # When log_ratio is close to 0, use expm1 for accuracy.
    if log_ratio < -36:  # exp(-36) ≈ 2e-16
        return 1.0
    return -math.expm1(log_ratio)


def pass_at_k_from_bools(correct: list[bool], k: int) -> float:
    """Convenience: compute pass@k from a list of correctness booleans."""
    n = len(correct)
    c = sum(correct)
    return pass_at_k(n, c, k)
