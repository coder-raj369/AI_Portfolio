"""Verifiers — deterministic scoring for checkable outputs.

Math: exact string match, numeric tolerance, and expression equivalence.
Code: execution-based (sandboxed) and static-analysis based.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any


def exact_match(pred: str, gold: str, *, strip: bool = True) -> bool:
    """Exact string match, optionally stripping whitespace."""
    if strip:
        pred = pred.strip()
        gold = gold.strip()
    return pred == gold


def numeric_match(
    pred: str, gold: str, *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Numeric match with tolerance.

    Tries to extract the *last* number from each string (common pattern:
    model outputs reasoning then "The answer is 42.").
    """
    pred_num = _extract_last_number(pred)
    gold_num = _extract_last_number(gold)
    if pred_num is None or gold_num is None:
        return False
    return math.isclose(pred_num, gold_num, rel_tol=rtol, abs_tol=atol)


def _extract_last_number(text: str) -> float | None:
    """Extract the last number-like token from text."""
    # Match integers, decimals, scientific notation, negative numbers
    numbers = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def expression_match(pred: str, gold: str) -> bool:
    """Check if two simple arithmetic expressions evaluate to the same value.

    Only allows literals and + - * / ** operators. Unsafe inputs are rejected.
    """
    try:
        pred_val = _safe_eval(pred)
        gold_val = _safe_eval(gold)
    except (ValueError, SyntaxError, ZeroDivisionError):
        return False
    if pred_val is None or gold_val is None:
        return False
    return math.isclose(pred_val, gold_val, rel_tol=1e-9)


def _safe_eval(expr: str) -> float | None:
    """Safely evaluate a simple numeric expression."""
    expr = expr.strip()
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node, rejecting anything unsafe."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Non-numeric constant")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported operator")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator")
    raise ValueError("Unsupported expression type")


class CodeVerifier:
    """Execution-based code verifier with a simple timeout sandbox.

    Runs Python code in a restricted namespace and checks the output or
    a returned value against an expected result.
    """

    def __init__(self, timeout_sec: float = 2.0) -> None:
        self.timeout_sec = timeout_sec

    def verify(
        self, code: str, test_fn: Callable[[Any], bool], entry_point: str = "solution"
    ) -> bool:
        """Run code and apply test_fn to the result.

        Args:
            code: Python source string.
            test_fn: Callable that takes the result and returns True/False.
            entry_point: Name of the function to call after executing the code.
                         If the code defines this function, it is extracted and
                         passed to test_fn.

        Returns:
            True if test_fn returns True, False otherwise (including on
            timeout, syntax error, or runtime exception).
        """
        import signal

        # Restricted globals
        safe_globals = {
            "__builtins__": {
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "round": round,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "tuple": tuple,
                "dict": dict,
                "set": set,
                "print": print,
                "True": True,
                "False": False,
                "None": None,
            }
        }
        local_ns: dict[str, Any] = {}

        try:
            compiled = compile(code, "<sandbox>", "exec")
        except SyntaxError:
            return False

        def _alarm_handler(_signum: int, _frame: Any) -> None:
            raise TimeoutError("Code execution timed out")

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout_sec)

        try:
            exec(compiled, safe_globals, local_ns)  # noqa: S102
            result = local_ns.get(entry_point)
            return test_fn(result)
        except Exception:
            return False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


# Import here to avoid circular issues with typing
from typing import Callable  # noqa: E402
