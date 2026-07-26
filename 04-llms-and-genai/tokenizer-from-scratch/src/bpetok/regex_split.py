r"""Pre-tokenization: split text before BPE ever sees it.

Without this step BPE happily learns merges that span word boundaries and
punctuation (`". The"` as one token), which wastes vocabulary and makes the
tokenizer sensitive to spacing. The pattern below is the GPT-4 / `cl100k_base`
shape:

* contractions are kept attached (`'s`, `'ve`, ...),
* a leading space joins the following word (` word`), which is why most tokens
  in a modern vocabulary start with a space,
* digits are grouped in runs of at most 3, so numbers tokenize consistently
  regardless of magnitude,
* runs of whitespace stay together.
"""

from __future__ import annotations

import regex  # the `re` module cannot do \p{L} character classes

GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|"""
    r"""[^\r\n\p{L}\p{N}]?+\p{L}+|"""
    r"""\p{N}{1,3}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|"""
    r"""\s*[\r\n]|"""
    r"""\s+(?!\S)|"""
    r"""\s+"""
)

_COMPILED = regex.compile(GPT4_SPLIT_PATTERN)


def pretokenize(text: str, pattern: str | None = None) -> list[str]:
    """Split `text` into pre-token chunks that BPE merges cannot cross."""
    compiled = _COMPILED if pattern is None else regex.compile(pattern)
    return compiled.findall(text)
