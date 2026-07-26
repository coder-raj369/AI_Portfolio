"""Training behaviour: determinism, vocabulary accounting, compression."""

from __future__ import annotations

import pytest

from bpetok import BPETokenizer, train_bpe

from corpora import CORPUS, SPECIALS


def test_training_is_deterministic() -> None:
    a = train_bpe(CORPUS, 500, special_tokens=SPECIALS)
    b = train_bpe(CORPUS, 500, special_tokens=SPECIALS)
    assert a.merges == b.merges
    assert a.vocab == b.vocab


def test_vocab_size_accounting_is_exact(tokenizer: BPETokenizer) -> None:
    # 256 bytes + merges + specials, and nothing else
    assert tokenizer.vocab_size == 256 + len(tokenizer.merges) + len(SPECIALS)
    # vocab_size is a ceiling, not a promise: this corpus is repetitive enough
    # that BPE runs out of pairs occurring twice before reaching 600 tokens.
    assert tokenizer.vocab_size <= 600
    assert tokenizer.vocab_size == 415


def test_special_tokens_get_the_highest_ids(tokenizer: BPETokenizer) -> None:
    ids = sorted(tokenizer.special_tokens.values())
    assert ids == list(range(256 + len(tokenizer.merges), tokenizer.vocab_size))


def test_merge_ids_are_contiguous_and_ordered() -> None:
    result = train_bpe(CORPUS, 400)
    assert list(result.merges.values()) == list(range(256, 256 + len(result.merges)))


def test_vocab_smaller_than_the_byte_alphabet_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than 256"):
        train_bpe(CORPUS, 100)


def test_training_stops_early_when_no_pair_repeats() -> None:
    """Asking for 5,000 merges from a 3-word corpus must not fabricate them."""
    result = train_bpe("ab ab ab", vocab_size=5000)
    assert len(result.merges) < 20
    assert result.stats["merges_learned"] == len(result.merges)


def test_compression_beats_byte_level(tokenizer: BPETokenizer) -> None:
    ratio = tokenizer.compression_ratio(CORPUS)
    assert ratio > 3.0, f"only {ratio:.2f} bytes/token"


def test_larger_vocabulary_compresses_better() -> None:
    ratios = []
    for size in (300, 500, 1000):
        tok = BPETokenizer.train(CORPUS, vocab_size=size)
        ratios.append(tok.compression_ratio(CORPUS))
    assert ratios == sorted(ratios), ratios
    assert ratios[-1] > ratios[0] * 1.3


def test_compression_on_held_out_text_is_lower_than_on_training_text() -> None:
    """Methodological check: quoting the in-sample number overstates the win."""
    train, held_out = CORPUS[: int(len(CORPUS) * 0.8)], CORPUS[int(len(CORPUS) * 0.8) :]
    tok = BPETokenizer.train(train, vocab_size=600)
    assert tok.compression_ratio(train) > tok.compression_ratio(held_out)


def test_learned_tokens_are_real_substrings_of_the_corpus(tokenizer: BPETokenizer) -> None:
    corpus_bytes = CORPUS.encode("utf-8")
    for tid in range(256, 256 + len(tokenizer.merges)):
        assert tokenizer.vocab[tid] in corpus_bytes


def test_stats_are_reported() -> None:
    result = train_bpe(CORPUS, 500)
    assert result.stats["corpus_bytes"] == len(CORPUS.encode("utf-8"))
    assert result.stats["unique_pretokens"] > 10
