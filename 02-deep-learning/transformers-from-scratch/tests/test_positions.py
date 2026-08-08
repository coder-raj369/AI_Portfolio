"""RoPE and ALiBi are each defined by a property, so each gets a property test."""

from __future__ import annotations

import numpy as np
import pytest

from nanoformer.reference import alibi_bias, alibi_slopes, apply_rope, rope_frequencies

RNG = np.random.default_rng(0)


def test_rope_logit_depends_only_on_relative_position() -> None:
    """The entire justification for RoPE: q_m . k_n is a function of (m - n).

    Checked across three absolute position pairs with the same offset. If the
    rotation were applied per-channel-pair incorrectly, absolute position would leak
    and these three numbers would differ.
    """
    cos, sin = rope_frequencies(16, 64)
    q, k = RNG.standard_normal((1, 1, 1, 16)), RNG.standard_normal((1, 1, 1, 16))

    def logit(m: int, n: int) -> float:
        return float(
            (apply_rope(q, cos, sin, offset=m) * apply_rope(k, cos, sin, offset=n)).sum()
        )

    base = logit(5, 3)
    assert logit(9, 7) == pytest.approx(base, rel=1e-12)
    assert logit(20, 18) == pytest.approx(base, rel=1e-12)
    assert logit(6, 3) != pytest.approx(base, rel=1e-6)  # different offset, different logit


def test_rope_is_a_rotation_so_it_preserves_norm() -> None:
    cos, sin = rope_frequencies(32, 128)
    x = RNG.standard_normal((2, 4, 10, 32))
    rotated = apply_rope(x, cos, sin, offset=13)
    np.testing.assert_allclose(
        np.linalg.norm(rotated, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-12
    )


def test_rope_at_position_zero_is_the_identity() -> None:
    cos, sin = rope_frequencies(16, 8)
    x = RNG.standard_normal((1, 1, 1, 16))
    np.testing.assert_allclose(apply_rope(x, cos, sin, offset=0), x, rtol=1e-12)


def test_rope_offset_matters_for_cached_decoding() -> None:
    """Encoding token 7 with offset=0 (the classic cache bug) is not the same as
    encoding it at its true position."""
    cos, sin = rope_frequencies(16, 32)
    x = RNG.standard_normal((1, 1, 1, 16))
    assert not np.allclose(apply_rope(x, cos, sin, offset=7), apply_rope(x, cos, sin, offset=0))


def test_rope_rejects_odd_head_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        rope_frequencies(15, 8)


def test_larger_theta_slows_the_lowest_frequency() -> None:
    """RoPE scaling in one assertion: raising theta shrinks the angle swept at a
    given position, which is how a model trained short is stretched to long context."""
    _, sin_small = rope_frequencies(64, 2048, theta=10_000.0)
    _, sin_large = rope_frequencies(64, 2048, theta=1_000_000.0)
    assert abs(sin_large[2047, -1]) < abs(sin_small[2047, -1])


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8, 16])
def test_alibi_slopes_are_a_decreasing_geometric_ladder(n_heads: int) -> None:
    slopes = alibi_slopes(n_heads)
    assert slopes.shape == (n_heads,)
    assert np.all(slopes > 0)
    assert np.all(np.diff(slopes) <= 0)
    if n_heads >= 8:  # 2^(-8k/n) ladder: first slope is 2^(-8/n)
        assert slopes[0] == pytest.approx(2.0 ** (-8.0 / n_heads), rel=1e-9)


@pytest.mark.parametrize("n_heads", [3, 5, 6, 12])
def test_alibi_handles_non_power_of_two_head_counts(n_heads: int) -> None:
    slopes = alibi_slopes(n_heads)
    assert slopes.shape == (n_heads,)
    assert np.all(slopes > 0) and np.all(np.isfinite(slopes))


def test_alibi_bias_is_zero_on_the_diagonal_and_negative_away_from_it() -> None:
    bias = alibi_bias(4, 6, 6)
    assert bias.shape == (4, 6, 6)
    np.testing.assert_allclose(np.diagonal(bias, axis1=1, axis2=2), 0.0, atol=0)
    assert (bias[:, 0, 5] < 0).all()


def test_alibi_bias_is_exactly_linear_in_distance() -> None:
    """Not "roughly decaying" - exactly -slope * |m - n|, which is checkable."""
    n_heads = 4
    bias = alibi_bias(n_heads, 8, 8)
    slopes = alibi_slopes(n_heads)
    for h in range(n_heads):
        for distance in (1, 3, 7):
            assert bias[h, distance, 0] == pytest.approx(-slopes[h] * distance, rel=1e-12)


def test_alibi_offset_shifts_the_query_positions() -> None:
    """Decoding token 10 against a 10-token history must see distances 10..1."""
    bias = alibi_bias(2, 1, 11, offset=10)
    slopes = alibi_slopes(2)
    assert bias[0, 0, 0] == pytest.approx(-slopes[0] * 10, rel=1e-12)
    assert bias[0, 0, 10] == pytest.approx(0.0)
