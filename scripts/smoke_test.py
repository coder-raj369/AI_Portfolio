"""Import and exercise every shipped module, so rot is caught immediately.

This is deliberately fast (< 5 s) and dependency-light: it is the check that runs
in CI on every push and the one to run first after cloning. It asserts on
*behaviour*, not just imports - an import-only smoke test passes happily while
the maths underneath is broken.

Run: `python scripts/smoke_test.py`
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each project keeps a src/ layout; add them all so this script works from a bare
# checkout without installing anything.
for src in sorted(ROOT.glob("*/*/src")):
    sys.path.insert(0, str(src))

CHECKS: list[tuple[str, object]] = []


def check(name: str):
    def register(fn):
        CHECKS.append((name, fn))
        return fn

    return register


@check("00-foundations/micrograd-engine")
def _nanograd() -> str:
    import numpy as np
    from nanograd import MLP, Tensor, cross_entropy, gradcheck, make_spirals

    a = Tensor(np.random.default_rng(0).standard_normal((3, 4)), requires_grad=True)
    err = gradcheck(lambda: (a * a).tanh().sum(), [a])
    assert err < 1e-5, err

    X, y = make_spirals(n_per_class=30, seed=0)
    model = MLP([2, 8, 3], seed=0)
    loss = cross_entropy(model(Tensor(X)), y)
    model.zero_grad()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    return f"gradcheck err={err:.1e}, loss={loss.item():.3f}"


@check("00-foundations/optimizers-from-scratch")
def _nanoptim() -> str:
    import numpy as np
    from nanoptim import SGD, Adam, AdamW, CosineWithWarmup, Lion, Parameter

    results = {}
    for name, build in (("sgd", lambda p: SGD(p, lr=0.1, momentum=0.9)),
                        ("adam", lambda p: Adam(p, lr=0.1)),
                        ("adamw", lambda p: AdamW(p, lr=0.1)),
                        ("lion", lambda p: Lion(p, lr=0.05))):
        p = Parameter(np.array([3.0, -2.0]))
        opt = build([p])
        for _ in range(200):
            p.grad = 2.0 * p.data
            opt.step()
        results[name] = float(np.abs(p.data).max())
        assert results[name] < 0.5, (name, results[name])
    sched = CosineWithWarmup(1e-3, total_steps=100, warmup_steps=10)
    assert sched(9) == 1e-3 and sched(100) == 0.0
    return "4 optimisers converge, schedule endpoints correct"


@check("00-foundations/nn-primitives")
def _nnprim() -> str:
    import numpy as np
    from nnprim import (
        layer_norm,
        layer_norm_backward,
        multi_head_attention,
        naive_multi_head_attention,
        numeric_grad,
        softmax,
    )

    big = np.array([[1000.0, 999.0, 998.0]])
    assert np.isfinite(softmax(big)).all()
    assert abs(float(softmax(big).sum()) - 1.0) < 1e-12

    rng = np.random.default_rng(0)
    x, g = rng.standard_normal((3, 8)), rng.standard_normal((3, 8))
    _, cache = layer_norm(x)
    dx, _, _ = layer_norm_backward(g, cache)
    err = float(np.abs(dx - numeric_grad(lambda: float((layer_norm(x)[0] * g).sum()), x)).max())
    assert err < 1e-7, err

    xb = rng.standard_normal((2, 5, 16))
    w = [rng.standard_normal((16, 16)) * 0.25 for _ in range(4)]
    fast, _ = multi_head_attention(xb, *w, n_heads=4, causal=True)
    slow = naive_multi_head_attention(xb, *w, n_heads=4, causal=True)
    diff = float(np.abs(fast - slow).max())
    assert diff < 1e-12, diff
    return f"softmax stable at logit 1000, LN grad err={err:.1e}, MHA vs naive={diff:.1e}"


@check("04-llms-and-genai/tokenizer-from-scratch")
def _bpetok() -> str:
    from bpetok import BPETokenizer

    corpus = "the quick brown fox jumps over the lazy dog " * 60 + "日本語 🚀 " * 20
    tok = BPETokenizer.train(corpus, vocab_size=400, special_tokens=["<|endoftext|>"])
    text = "the quick 🚀 日本語 <|endoftext|>"
    assert tok.decode(tok.encode(text)) == text
    assert tok.encode("<|endoftext|>", allowed_special="all") == [
        tok.special_tokens["<|endoftext|>"]
    ]
    ratio = tok.compression_ratio(corpus)
    assert ratio > 2.0, ratio
    return f"vocab={tok.vocab_size}, lossless roundtrip, {ratio:.2f} bytes/token"


def main() -> int:
    width = max(len(name) for name, _ in CHECKS)
    failures = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
            print(f"PASS  {name:<{width}}  {detail}")
        except Exception:
            failures += 1
            print(f"FAIL  {name:<{width}}")
            traceback.print_exc()
    total = len(CHECKS)
    print(f"\n{total - failures}/{total} modules healthy")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
