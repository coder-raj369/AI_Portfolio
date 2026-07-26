"""Byte-level BPE must be lossless on arbitrary input, with no <unk> token."""

from __future__ import annotations

import random

import pytest

from bpetok import BPETokenizer

from corpora import CORPUS, SPECIALS

TRICKY = [
    "",
    " ",
    "\n\n\t  \r\n",
    "hello world",
    "Hello, World! 123",
    "日本語のテキストです",
    "emoji 🚀🐈‍⬛👩🏽‍💻 family 👨‍👩‍👧‍👦",
    "combining: e\u0301 a\u030a n\u0303",
    "zero width\u200bjoiner\u200d here",
    "math ∑∫∂√∞ ≠ ≤",
    "RTL: العربية and עברית",
    "control chars: \x00\x01\x1f\x7f",
    "surrogate-free astral: 𝕳𝖊𝖑𝖑𝖔 𝓦𝓸𝓻𝓵𝓭",
    "mixed العربية 日本 🚀 code: `x = f(1,2)`",
    "a" * 500,
]


@pytest.mark.parametrize("text", TRICKY)
def test_roundtrip_is_lossless(tokenizer: BPETokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_on_random_unicode_fuzz(tokenizer: BPETokenizer) -> None:
    """1,000 random strings drawn from across the BMP plus astral planes."""
    rng = random.Random(1234)
    for _ in range(1000):
        length = rng.randint(0, 40)
        chars = []
        for _ in range(length):
            code = rng.choice(
                [rng.randint(1, 0x7F), rng.randint(0x80, 0x7FF), rng.randint(0x800, 0xD7FF),
                 rng.randint(0xE000, 0xFFFF), rng.randint(0x10000, 0x10FFFF)]
            )
            chars.append(chr(code))
        text = "".join(chars)
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_every_byte_value_is_representable(tokenizer: BPETokenizer) -> None:
    """The base vocabulary is all 256 bytes, so nothing can be out-of-vocabulary."""
    assert all(i in tokenizer.vocab for i in range(256))
    raw = bytes(range(256))
    text = raw.decode("latin-1")
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_decoding_a_partial_multibyte_character_does_not_crash(
    tokenizer: BPETokenizer,
) -> None:
    """Streaming generation emits token prefixes: a lone continuation byte is a
    legal token sequence and must degrade, not raise."""
    ids = tokenizer.encode("日本語")
    partial = tokenizer.decode(ids[:1])
    assert isinstance(partial, str)


def test_decode_rejects_an_unknown_id(tokenizer: BPETokenizer) -> None:
    with pytest.raises(ValueError, match="unknown token id"):
        tokenizer.decode([999_999])


def test_special_tokens_roundtrip_when_allowed(tokenizer: BPETokenizer) -> None:
    text = "<|im_start|>user\nhi<|im_end|><|endoftext|>"
    ids = tokenizer.encode(text, allowed_special="all")
    assert tokenizer.decode(ids) == text
    # each special token collapses to exactly one id
    assert sum(1 for i in ids if i in tokenizer.special_tokens.values()) == 3


def test_untrained_text_still_roundtrips(tokenizer: BPETokenizer) -> None:
    """Nothing in this string appears in the training corpus."""
    text = "Zqx vburgle 9182 ∮ 𝔯𝔞𝔯𝔢"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_holds_for_a_freshly_trained_tiny_vocab() -> None:
    tiny = BPETokenizer.train(CORPUS[:2000], vocab_size=260, special_tokens=SPECIALS)
    assert tiny.decode(tiny.encode("anything at all 🚀")) == "anything at all 🚀"
