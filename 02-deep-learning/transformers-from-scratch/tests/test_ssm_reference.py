"""The selective scan: two implementations, one answer, plus the S4 contrast."""

from __future__ import annotations

import numpy as np
import pytest

from nanoformer.reference import (
    SelectiveSSMBlock,
    lti_scan_via_convolution,
    selective_scan_chunked,
    selective_scan_sequential,
    softplus,
)

RNG = np.random.default_rng(3)


def _inputs(batch=2, seq=40, d_inner=6, d_state=8):
    x = RNG.standard_normal((batch, seq, d_inner))
    delta = softplus(RNG.standard_normal((batch, seq, d_inner)))
    A = -np.tile(np.arange(1, d_state + 1, dtype=np.float64), (d_inner, 1))
    B = RNG.standard_normal((batch, seq, d_state))
    C = RNG.standard_normal((batch, seq, d_state))
    D = np.ones(d_inner)
    return x, delta, A, B, C, D


@pytest.mark.parametrize("chunk", [1, 4, 16, 64])
def test_chunked_scan_matches_the_sequential_recurrence(chunk: int) -> None:
    """The headline correctness claim. A chunked/parallel formulation is only useful
    if it is *the same function* as the recurrence - including across chunk
    boundaries, which is where the carried state is easy to drop."""
    x, delta, A, B, C, D = _inputs()
    seq_out = selective_scan_sequential(x, delta, A, B, C, D)
    chunk_out = selective_scan_chunked(x, delta, A, B, C, D, chunk=chunk)
    assert np.abs(seq_out - chunk_out).max() < 1e-10


def test_chunk_size_larger_than_sequence_is_handled() -> None:
    x, delta, A, B, C, D = _inputs(seq=7)
    a = selective_scan_sequential(x, delta, A, B, C, D)
    b = selective_scan_chunked(x, delta, A, B, C, D, chunk=1000)
    assert np.abs(a - b).max() < 1e-10


def test_scan_is_causal() -> None:
    """Perturbing the last input must not change any earlier output."""
    x, delta, A, B, C, D = _inputs(seq=20)
    base = selective_scan_sequential(x, delta, A, B, C, D)
    x2 = x.copy()
    x2[:, -1] += 100.0
    after = selective_scan_sequential(x2, delta, A, B, C, D)
    np.testing.assert_allclose(base[:, :-1], after[:, :-1], atol=1e-12)
    assert not np.allclose(base[:, -1], after[:, -1])


def test_negative_A_makes_the_recurrence_contract() -> None:
    """exp(delta * A) with A < 0 lies in (0, 1), so state decays instead of exploding.

    This is why A is initialised negative, and why parameterisations that *cannot*
    flip its sign (storing `A_log` and using `-exp(A_log)`) are preferred: a sign
    flip during training does not degrade the model, it destroys it.
    """
    x, delta, A, B, C, D = _inputs(seq=40)
    out = selective_scan_sequential(x, delta, A, B, C, D)
    assert np.all(np.isfinite(out))
    assert np.abs(out).max() < 1e3

    blown = selective_scan_sequential(x, delta, -A, B, C, D)  # positive A: amplifies
    assert np.abs(blown).max() > np.abs(out).max() * 1e6


def test_positive_A_overflows_outright_on_a_long_sequence() -> None:
    """Not merely "larger": at seq=200 the unstable recurrence leaves float range
    entirely and produces non-finite values."""
    x, delta, A, B, C, D = _inputs(seq=200)
    with np.errstate(over="ignore", invalid="ignore"):
        blown = selective_scan_sequential(x, delta, -A, B, C, D)
    assert not np.all(np.isfinite(blown))
    assert np.all(np.isfinite(selective_scan_sequential(x, delta, A, B, C, D)))


def test_large_delta_forgets_and_small_delta_remembers() -> None:
    """`delta` is the selectivity knob: it is the discretisation step, so a large
    delta drives exp(delta*A) towards 0 and wipes the state."""
    batch, seq, d_inner, d_state = 1, 30, 2, 4
    x = np.zeros((batch, seq, d_inner))
    x[:, 0] = 1.0  # a single impulse at t=0
    A = -np.ones((d_inner, d_state))
    B = np.ones((batch, seq, d_state))
    C = np.ones((batch, seq, d_state))

    forgetful = selective_scan_sequential(x, np.full_like(x, 3.0), A, B, C)
    retentive = selective_scan_sequential(x, np.full_like(x, 0.05), A, B, C)
    # Ratio of the response at t=20 to the response at t=0.
    assert abs(forgetful[0, 20, 0] / forgetful[0, 0, 0]) < 1e-20
    assert abs(retentive[0, 20, 0] / retentive[0, 0, 0]) > 0.3


def test_non_selective_ssm_reduces_to_a_convolution() -> None:
    """With constant A, B, C the recurrence is an FIR filter - the property S4
    exploits for O(L log L) training, and precisely what selectivity gives up."""
    x = RNG.standard_normal((3, 25))
    a, b, c = np.array(0.9), np.array(0.5), np.array(1.3)
    conv = lti_scan_via_convolution(x, a, b, c)

    h = np.zeros(3)
    recurrent = np.zeros_like(x)
    for t in range(25):
        h = a * h + b * x[:, t]
        recurrent[:, t] = c * h
    np.testing.assert_allclose(conv, recurrent, atol=1e-12)


def test_softplus_does_not_overflow_and_saturates_to_identity() -> None:
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    out = softplus(x)
    assert np.all(np.isfinite(out)) and np.all(out >= 0)
    assert out[-1] == pytest.approx(1000.0, rel=1e-9)  # saturates to identity
    assert out[2] == pytest.approx(np.log(2.0))


def test_softplus_underflows_to_exactly_zero_for_very_negative_input() -> None:
    """A real limitation, asserted rather than assumed away: softplus is positive in
    exact arithmetic but `log1p(exp(-1000))` rounds to exactly 0.0. A `delta` of 0
    freezes the SSM state (`exp(0*A) = 1`, input term 0), so a pathological dt
    projection can silently produce a layer that ignores its input. The `dt_bias`
    initialisation of -2 keeps delta near 0.13 and far from this regime."""
    assert softplus(np.array([-1000.0]))[0] == 0.0
    assert softplus(np.array([-2.0]))[0] == pytest.approx(0.1269, rel=1e-3)


def test_ssm_block_forward_and_scan_paths_agree() -> None:
    block = SelectiveSSMBlock(d_model=16, d_state=8, seed=0)
    u = RNG.standard_normal((2, 24, 16))
    a = block.forward(u, chunked=False)
    b = block.forward(u, chunked=True)
    assert a.shape == (2, 24, 16)
    assert np.abs(a - b).max() < 1e-10


def test_ssm_state_size_is_independent_of_sequence_length() -> None:
    """The structural advantage over attention: memory is O(d_state), not O(seq).
    A 4x longer sequence must not change the state footprint."""
    block = SelectiveSSMBlock(d_model=16, d_state=8, seed=0)
    short = block.forward(RNG.standard_normal((1, 16, 16)))
    long = block.forward(RNG.standard_normal((1, 64, 16)))
    assert short.shape[1] == 16 and long.shape[1] == 64
    state_elements = block.d_inner * block.d_state
    assert state_elements == 32 * 8  # expand=2 -> d_inner=32; independent of seq
