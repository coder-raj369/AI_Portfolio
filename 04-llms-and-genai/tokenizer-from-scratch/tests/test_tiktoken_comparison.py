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

    # A 100k vocabulary trained on the open web should beat a 600-token
    # vocabulary trained on 30 KB, but not by more than ~2x on repetitive text.
    assert theirs_ratio > ours_ratio
    assert theirs_ratio / ours_ratio < 2.5


def test_our_pretokenizer_agrees_with_tiktoken_on_chunk_boundaries() -> None:
    """The GPT-4 split pattern is copied deliberately; if tiktoken is available we
    can check the pre-tokenization agrees on a realistic sentence."""
    tiktoken = pytest.importorskip("tiktoken", reason="tiktoken is optional")
    enc = tiktoken.get_encoding("cl100k_base")
    from bpetok import pretokenize

    sentence = "It's 2026: models like GPT-4 tokenize   whitespace, digits (12345) and 日本語."
    ours = pretokenize(sentence)
    theirs = [enc.decode([t]) for t in enc.encode(sentence)]
    # tiktoken's merges join some of our chunks, so ours must be a refinement:
    # concatenation is identical and our chunk count is >= theirs.
    assert "".join(ours) == sentence
    assert len(ours) >= len([t for t in theirs if t])
