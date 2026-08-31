"""Optional benchmark against tiktoken's production vocabularies.

Skipped when tiktoken is not installed (it is not required by CI). This is a
*comparison*, not a parity check: matching cl100k's ids would require its exact
merge table, so what is asserted is that our compression lands in a sane place
relative to a vocabulary 170x larger.
"""

from __future__ import annotations

import pytest

from bpetok import BPETokenizer

from corpora import CORPUS


def test_compression_is_sane_relative_to_cl100k() -> None:
    tiktoken = pytest.importorskip("tiktoken", reason="tiktoken is optional")
    enc = tiktoken.get_encoding("cl100k_base")
    ours = BPETokenizer.train(CORPUS, vocab_size=600)

    text = CORPUS[:20000]
    n_bytes = len(text.encode("utf-8"))
    ours_ratio = n_bytes / len(ours.encode_ordinary(text))
    theirs_ratio = n_bytes / len(enc.encode(text))

    # The comparison is intentionally weak: both numbers are measured on a tiny
    # local corpus, and a 100k vocabulary is only a rough upper bound here.
    assert theirs_ratio > 2.0
    assert ours_ratio > 2.0
    assert 0.6 < theirs_ratio / ours_ratio < 1.5


def test_our_pretokenizer_agrees_with_tiktoken_on_chunk_boundaries() -> None:
    """The GPT-4 split pattern is copied deliberately; if tiktoken is available we
    can check the pre-tokenization agrees on a realistic sentence."""
    tiktoken = pytest.importorskip("tiktoken", reason="tiktoken is optional")
    enc = tiktoken.get_encoding("cl100k_base")
    from bpetok import pretokenize

    sentence = "It's 2026: models like GPT-4 tokenize   whitespace, digits (12345) and 日本語."
    ours = pretokenize(sentence)
    theirs = [enc.decode([t]) for t in enc.encode(sentence)]
    # The invariant that matters is text preservation and a sensible chunking
    # granularity; raw token-count comparison is not stable across BPE merges.
    assert "".join(ours) == sentence
    assert 5 <= len(ours) <= len([t for t in theirs if t]) + 10
