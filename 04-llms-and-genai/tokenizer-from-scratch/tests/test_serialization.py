"""Save/load must reproduce identical token ids, not merely similar ones."""

from __future__ import annotations

import json
from pathlib import Path

from bpetok import BPETokenizer

from corpora import CORPUS, SPECIALS


def test_save_load_reproduces_identical_ids(tokenizer: BPETokenizer, tmp_path=None) -> None:
    path = Path(tmp_path or ".") / "tok.json"
    tokenizer.save(path)
    reloaded = BPETokenizer.load(path)
    for text in ("hello world", "日本語 🚀", CORPUS[:400], "<|endoftext|>"):
        assert reloaded.encode(text) == tokenizer.encode(text)
        assert reloaded.encode(text, allowed_special="all") == tokenizer.encode(
            text, allowed_special="all"
        )
    path.unlink()


def test_loaded_vocab_is_rebuilt_exactly(tokenizer: BPETokenizer, tmp_path=None) -> None:
    path = Path(tmp_path or ".") / "tok2.json"
    tokenizer.save(path)
    reloaded = BPETokenizer.load(path)
    assert reloaded.vocab == tokenizer.vocab
    assert reloaded.special_tokens == tokenizer.special_tokens
    path.unlink()


def test_merges_are_serialised_as_an_ordered_list(tokenizer: BPETokenizer, tmp_path=None) -> None:
    """Order is semantically load-bearing, so it must not depend on dict/JSON
    object key ordering."""
    path = Path(tmp_path or ".") / "tok3.json"
    tokenizer.save(path)
    payload = json.loads(path.read_text())
    assert isinstance(payload["merges"], list)
    assert [m[2] for m in payload["merges"]] == sorted(m[2] for m in payload["merges"])
    path.unlink()


def test_roundtrip_through_disk_preserves_losslessness(tmp_path=None) -> None:
    tok = BPETokenizer.train(CORPUS[:5000], vocab_size=400, special_tokens=SPECIALS)
    path = Path(tmp_path or ".") / "tok4.json"
    tok.save(path)
    reloaded = BPETokenizer.load(path)
    text = "arbitrary text 🚀 with 日本語 and 12345"
    assert reloaded.decode(reloaded.encode(text)) == text
    path.unlink()
