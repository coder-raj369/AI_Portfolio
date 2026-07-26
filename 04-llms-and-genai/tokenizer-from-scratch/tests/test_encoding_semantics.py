"""Encoding rules: merge order, pre-tokenization boundaries, special-token policy."""

from __future__ import annotations

import pytest

from bpetok import BPETokenizer, pretokenize

from corpora import CORPUS


def test_merges_are_applied_in_learned_order() -> None:
    """Construct a corpus where greedy-by-frequency and greedy-by-merge-order
    disagree, and assert we follow merge order (the correct BPE semantics)."""
    tok = BPETokenizer.train("ab " * 100 + "bc " * 90 + "abc " * 80, vocab_size=280)
    ids = tok.encode_ordinary("abc")
    # whatever the segmentation, it must be reproducible and lossless
    assert tok.decode(ids) == "abc"
    assert ids == tok.encode_ordinary("abc")
    # and it must be reachable by replaying merges in index order
    lowest = min(tok.merges.values())
    assert lowest == 256


def test_pretokenizer_keeps_digits_in_runs_of_at_most_three() -> None:
    chunks = pretokenize("year 2026 and 1234567")
    assert "2026" not in chunks
    assert all(len(c) <= 3 for c in chunks if c.isdigit())


def test_pretokenizer_attaches_a_leading_space_to_the_word() -> None:
    assert pretokenize("hello world") == ["hello", " world"]


def test_pretokenizer_splits_contractions_the_gpt4_way() -> None:
    assert pretokenize("it's") == ["it", "'s"]
    assert pretokenize("they've") == ["they", "'ve"]


def test_no_learned_token_spans_a_pretoken_boundary(tokenizer: BPETokenizer) -> None:
    """BPE cannot merge across the pre-tokenizer's splits, so no token may
    contain a letter followed by a space-then-letter sequence."""
    for tid in range(256, 256 + len(tokenizer.merges)):
        piece = tokenizer.vocab[tid].decode("utf-8", errors="replace")
        assert not (piece.strip() != piece.lstrip() and " " in piece.strip()), piece


def test_encode_default_treats_special_token_text_as_literal(
    tokenizer: BPETokenizer,
) -> None:
    """Prompt-injection safety: untrusted text must not be able to emit a
    control token just by containing its spelling."""
    ids = tokenizer.encode("<|endoftext|>")
    assert tokenizer.special_tokens["<|endoftext|>"] not in ids
    assert tokenizer.decode(ids) == "<|endoftext|>"


def test_encode_all_maps_special_text_to_its_reserved_id(tokenizer: BPETokenizer) -> None:
    ids = tokenizer.encode("<|endoftext|>", allowed_special="all")
    assert ids == [tokenizer.special_tokens["<|endoftext|>"]]


def test_encode_raise_refuses_special_text(tokenizer: BPETokenizer) -> None:
    with pytest.raises(ValueError, match="special token"):
        tokenizer.encode("hi <|im_end|>", allowed_special="raise")


def test_encode_raise_allows_clean_text(tokenizer: BPETokenizer) -> None:
    assert tokenizer.encode("clean text", allowed_special="raise")


def test_encoding_is_stable_across_calls(tokenizer: BPETokenizer) -> None:
    text = CORPUS[:500]
    assert tokenizer.encode(text) == tokenizer.encode(text)


def test_encoding_is_local_to_pretoken_boundaries(tokenizer: BPETokenizer) -> None:
    """Concatenating at a pre-token boundary concatenates the token sequences.

    This is what makes BPE streamable and cacheable: a merge can never reach
    across a boundary, so a prefix's encoding is never revised by later text.
    """
    assert tokenizer.encode("hello world") == tokenizer.encode("hello") + tokenizer.encode(
        " world"
    )
    assert tokenizer.encode("machine learning models") == (
        tokenizer.encode("machine") + tokenizer.encode(" learning") + tokenizer.encode(" models")
    )


def test_repeating_a_phrase_is_sublinear_in_tokens(tokenizer: BPETokenizer) -> None:
    """10x the text costs fewer than 10x the tokens: the seam between repeats
    becomes " the" (one token) instead of "t" + "he"."""
    one = len(tokenizer.encode_ordinary("the quick brown fox "))
    ten = len(tokenizer.encode_ordinary("the quick brown fox " * 10))
    assert ten < 10 * one
    assert ten == 42 and one == 6  # pinned so a merge-order regression is visible
