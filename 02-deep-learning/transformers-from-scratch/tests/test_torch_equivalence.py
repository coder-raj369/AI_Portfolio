"""The PyTorch implementations must agree with the NumPy reference, numerically.

Skipped when `torch` is absent, which is how CI stays CPU-only and fast. These are
the tests that make the PyTorch code *verified* rather than merely plausible: the
reference is independently checked against closed-form properties in the other test
files, so agreement here transfers that confidence.

Every test runs on CPU in float64 where possible - float32 tolerances are loose
enough to hide a genuinely wrong transpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoformer.reference import alibi_bias as ref_alibi_bias
from nanoformer.reference import alibi_slopes as ref_alibi_slopes
from nanoformer.reference import apply_rope, rope_frequencies
from nanoformer.reference import selective_scan_sequential as ref_scan

RNG = np.random.default_rng(0)


def _torch():
    return pytest.importorskip("torch", reason="torch is optional for this repo's CPU CI")


def test_rotary_embedding_matches_reference() -> None:
    torch = _torch()
    from nanoformer.torch_impl import RotaryEmbedding

    head_dim, seq = 32, 20
    x = RNG.standard_normal((2, 4, seq, head_dim))
    cos, sin = rope_frequencies(head_dim, 64)
    expected = apply_rope(x, cos, sin, offset=5)

    rope = RotaryEmbedding(head_dim, max_seq=64).double()
    got = rope(torch.tensor(x, dtype=torch.float64), offset=5).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_rotary_embedding_rejects_positions_past_its_table() -> None:
    torch = _torch()
    from nanoformer.torch_impl import RotaryEmbedding

    rope = RotaryEmbedding(16, max_seq=8)
    with pytest.raises(ValueError, match="max_seq"):
        rope(torch.zeros(1, 1, 4, 16), offset=6)


def test_alibi_slopes_and_bias_match_reference() -> None:
    _torch()
    from nanoformer.torch_impl import alibi_bias, alibi_slopes

    for n_heads in (1, 4, 8, 12, 16):
        np.testing.assert_allclose(
            alibi_slopes(n_heads, dtype=__import__("torch").float64).numpy(),
            ref_alibi_slopes(n_heads),
            rtol=1e-12,
        )
    import torch as _t

    np.testing.assert_allclose(
        alibi_bias(4, 6, 9, offset=3, dtype=_t.float64).numpy(),
        ref_alibi_bias(4, 6, 9, offset=3),
        rtol=1e-12,
    )


def _sync_attention(torch_attn, ref_attn) -> None:
    """Copy reference weights into the torch module.

    `nn.Linear` computes `x @ W.T`, the reference computes `x @ W`, so every weight
    is transposed on the way in. Getting this backwards produces plausible-looking
    output that is simply a different function - which is the whole reason this test
    exists.
    """
    import torch

    with torch.no_grad():
        torch_attn.w_q.weight.copy_(torch.tensor(ref_attn.w_q.T))
        torch_attn.w_k.weight.copy_(torch.tensor(ref_attn.w_k.T))
        torch_attn.w_v.weight.copy_(torch.tensor(ref_attn.w_v.T))
        torch_attn.w_o.weight.copy_(torch.tensor(ref_attn.w_o.T))


@pytest.mark.parametrize("n_kv_heads", [1, 2, 8])
def test_grouped_query_attention_matches_reference(n_kv_heads: int) -> None:
    torch = _torch()
    from nanoformer.reference import GroupedQueryAttention as RefGQA
    from nanoformer.torch_impl import GroupedQueryAttention as TorchGQA

    d_model, n_heads, seq = 64, 8, 10
    ref = RefGQA(d_model, n_heads, n_kv_heads, max_seq=64, seed=0)
    mod = TorchGQA(d_model, n_heads, n_kv_heads, max_seq=64).double()
    _sync_attention(mod, ref)

    x = RNG.standard_normal((2, seq, d_model))
    expected = ref.forward(x, causal=True)
    got = mod(torch.tensor(x, dtype=torch.float64), causal=True).detach().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_torch_kv_cache_matches_torch_full_forward() -> None:
    """Same equivalence the reference asserts, now for the PyTorch path - because a
    cache bug can live in either implementation independently."""
    torch = _torch()
    from nanoformer.torch_impl import GroupedQueryAttention, KVCache

    mod = GroupedQueryAttention(64, 8, 2, max_seq=64).double().eval()
    x = torch.tensor(RNG.standard_normal((1, 12, 64)), dtype=torch.float64)
    with torch.no_grad():
        full = mod(x, causal=True)
        cache = KVCache(1, mod.n_kv_heads, mod.head_dim, 64, dtype=torch.float64)
        steps = [mod(x[:, t : t + 1], cache=cache, causal=True) for t in range(12)]
    np.testing.assert_allclose(
        torch.cat(steps, dim=1).numpy(), full.numpy(), atol=1e-10
    )


def test_mha_matches_torch_scaled_dot_product_attention() -> None:
    """Cross-check against the library kernel: with RoPE off and n_kv == n_heads, our
    hand-written attention must equal `F.scaled_dot_product_attention`."""
    torch = _torch()
    from nanoformer.torch_impl import GroupedQueryAttention

    mod = GroupedQueryAttention(64, 8, 8, max_seq=64).double().eval()
    x = torch.tensor(RNG.standard_normal((2, 9, 64)), dtype=torch.float64)
    with torch.no_grad():
        ours = mod(x, causal=True, use_rope=False)

        b, s, _ = x.shape
        q = mod.w_q(x).view(b, s, 8, 8).transpose(1, 2)
        k = mod.w_k(x).view(b, s, 8, 8).transpose(1, 2)
        v = mod.w_v(x).view(b, s, 8, 8).transpose(1, 2)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = mod.w_o(ref.transpose(1, 2).reshape(b, s, 64))
    # -1e9 masking (not -inf) leaves a ~1e-9 relative difference vs a true -inf mask.
    np.testing.assert_allclose(ours.numpy(), ref.numpy(), atol=1e-7)


def test_rmsnorm_matches_the_closed_form() -> None:
    torch = _torch()
    from nanoformer.torch_impl import RMSNorm

    x = RNG.standard_normal((3, 7, 16))
    norm = RMSNorm(16, eps=1e-6).double()
    got = norm(torch.tensor(x, dtype=torch.float64)).detach().numpy()
    expected = x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_selective_scan_matches_reference() -> None:
    torch = _torch()
    from nanoformer.torch_impl import selective_scan_sequential

    b, s, i, n = 2, 24, 6, 8
    x = RNG.standard_normal((b, s, i))
    delta = np.log1p(np.exp(RNG.standard_normal((b, s, i))))
    A = -np.tile(np.arange(1, n + 1, dtype=np.float64), (i, 1))
    B, C = RNG.standard_normal((b, s, n)), RNG.standard_normal((b, s, n))
    D = np.ones(i)

    expected = ref_scan(x, delta, A, B, C, D)
    got = selective_scan_sequential(
        *(torch.tensor(t, dtype=torch.float64) for t in (x, delta, A, B, C, D))
    ).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-10)


@pytest.mark.parametrize("chunk", [1, 8, 32])
def test_torch_chunked_scan_matches_torch_sequential_scan(chunk: int) -> None:
    torch = _torch()
    from nanoformer.torch_impl import selective_scan_chunked, selective_scan_sequential

    b, s, i, n = 2, 20, 4, 6
    args = [
        torch.tensor(RNG.standard_normal((b, s, i)), dtype=torch.float64),
        torch.tensor(np.log1p(np.exp(RNG.standard_normal((b, s, i)))), dtype=torch.float64),
        torch.tensor(-np.tile(np.arange(1, n + 1, dtype=np.float64), (i, 1))),
        torch.tensor(RNG.standard_normal((b, s, n)), dtype=torch.float64),
        torch.tensor(RNG.standard_normal((b, s, n)), dtype=torch.float64),
        torch.ones(i, dtype=torch.float64),
    ]
    seq_out = selective_scan_sequential(*args)
    chunk_out = selective_scan_chunked(*args, chunk=chunk)
    np.testing.assert_allclose(chunk_out.numpy(), seq_out.numpy(), atol=1e-10)


def test_moe_auxiliary_loss_is_one_at_uniform_routing() -> None:
    """Same calibration property the reference asserts, on the torch module."""
    torch = _torch()
    from nanoformer.torch_impl import MoEFeedForward

    moe = MoEFeedForward(32, 64, n_experts=8, k=2).double()
    with torch.no_grad():
        moe.router.weight.zero_()  # all logits equal -> perfectly uniform probs
    x = torch.tensor(RNG.standard_normal((4, 32, 32)), dtype=torch.float64)
    _, stats = moe(x)
    assert float(stats.load_balancing_loss) == pytest.approx(1.0, rel=1e-6)
    assert stats.dead_experts == 0


def test_moe_reports_dropped_tokens_under_capacity_pressure() -> None:
    torch = _torch()
    from nanoformer.torch_impl import MoEFeedForward

    x = torch.tensor(RNG.standard_normal((4, 64, 32)), dtype=torch.float64)
    limited = MoEFeedForward(32, 64, n_experts=8, k=2, capacity_factor=0.25).double()
    _, stats = limited(x)
    assert stats.tokens_dropped > 0

    unlimited = MoEFeedForward(32, 64, n_experts=8, k=2).double()
    _, stats2 = unlimited(x)
    assert stats2.tokens_dropped == 0


def test_moe_gradients_reach_every_expert_via_the_auxiliary_loss() -> None:
    """A dead expert receives no gradient from the task loss. The auxiliary loss is
    what keeps its router column learning - so it must be differentiable and it must
    actually be in the graph."""
    torch = _torch()
    from nanoformer.torch_impl import MoEFeedForward

    moe = MoEFeedForward(16, 32, n_experts=4, k=1).double()
    x = torch.tensor(RNG.standard_normal((2, 8, 16)), dtype=torch.float64)
    out, stats = moe(x)
    (out.sum() + moe.auxiliary_loss(stats)).backward()
    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()
    assert moe.router.weight.grad.abs().sum() > 0


def test_model_forward_backward_and_shapes() -> None:
    torch = _torch()
    from nanoformer.torch_impl import ModelConfig, TinyDecoderLM

    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq=32, layer_pattern=("attn", "attn"))
    model = TinyDecoderLM(cfg)
    tokens = torch.randint(0, 64, (2, 10))
    logits, aux = model(tokens)
    assert logits.shape == (2, 10, 64)
    assert float(aux) == 0.0  # no MoE layers
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 64), tokens[:, 1:].reshape(-1)
    )
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )


def test_hybrid_and_moe_model_configurations_run() -> None:
    torch = _torch()
    from nanoformer.torch_impl import ModelConfig, TinyDecoderLM

    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=4, n_heads=4, n_kv_heads=2,
                      max_seq=32, layer_pattern=("ssm", "ssm", "attn", "ssm"),
                      moe_layers=(1, 3), n_experts=4, moe_k=2)
    model = TinyDecoderLM(cfg)
    logits, aux = model(torch.randint(0, 64, (2, 8)))
    assert logits.shape == (2, 8, 64)
    assert float(aux) > 0.0  # MoE layers contribute an auxiliary term
    # MoE inflates total parameters but not the per-token cost.
    assert model.active_parameters_per_token() < model.num_parameters()


def test_model_config_validates_its_inputs() -> None:
    _torch()
    from nanoformer.torch_impl import ModelConfig

    with pytest.raises(ValueError, match="layer_pattern"):
        ModelConfig(n_layers=3, layer_pattern=("attn", "attn"))
    with pytest.raises(ValueError, match="unknown mixer"):
        ModelConfig(n_layers=2, layer_pattern=("attn", "conv"))
    with pytest.raises(ValueError, match="moe_layers"):
        ModelConfig(n_layers=2, layer_pattern=("attn", "attn"), moe_layers=(5,))


def test_cached_generation_matches_uncached_greedy_decoding() -> None:
    """The serving-path equivalence: generating with a KV cache must produce the same
    tokens as re-running the full prefix each step. Verified with temperature->0
    (argmax) so sampling noise cannot mask a cache bug."""
    torch = _torch()
    from nanoformer.torch_impl import ModelConfig, TinyDecoderLM

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq=32, layer_pattern=("attn", "attn"))
    model = TinyDecoderLM(cfg).eval()
    prompt = torch.randint(0, 32, (1, 4))

    with torch.no_grad():
        cached = model.generate(prompt, max_new_tokens=6, temperature=1e-8)
        tokens = prompt.clone()
        for _ in range(6):
            logits, _ = model(tokens)
            tokens = torch.cat([tokens, logits[:, -1:].argmax(dim=-1)], dim=1)
    assert cached.tolist() == tokens.tolist()
