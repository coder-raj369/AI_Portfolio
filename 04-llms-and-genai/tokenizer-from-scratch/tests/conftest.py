"""Shared fixtures: one trained tokenizer, reused across the test modules."""

from __future__ import annotations

import pytest
from corpora import CORPUS, SPECIALS

from bpetok import BPETokenizer


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=600, special_tokens=SPECIALS)
