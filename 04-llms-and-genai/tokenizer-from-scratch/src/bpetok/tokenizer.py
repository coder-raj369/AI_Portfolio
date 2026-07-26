"""The tokenizer itself: encode, decode, special tokens, save/load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import regex

from bpetok.regex_split import pretokenize
from bpetok.trainer import TrainResult, train_bpe

SpecialPolicy = Literal["all", "none", "raise"]


class BPETokenizer:
    """A byte-level BPE tokenizer.

    Attributes:
        merges: ordered `(a, b) -> new_id` map.
        vocab: `id -> bytes`.
        special_tokens: `str -> id`.
    """

    def __init__(
        self,
        merges: dict[tuple[int, int], int],
        vocab: dict[int, bytes],
        special_tokens: dict[str, int] | None = None,
        pattern: str | None = None,
    ) -> None:
        self.merges = merges
        self.vocab = vocab
        self.special_tokens = special_tokens or {}
        self.pattern = pattern
        self._inverse_special = {v: k for k, v in self.special_tokens.items()}

    # ------------------------------------------------------------------ build
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        pattern: str | None = None,
        verbose: bool = False,
    ) -> BPETokenizer:
        result: TrainResult = train_bpe(
            text, vocab_size, special_tokens=special_tokens, pattern=pattern, verbose=verbose
        )
        specials = {}
        if special_tokens:
            first = 256 + len(result.merges)
            specials = {tok: first + i for i, tok in enumerate(special_tokens)}
        tok = cls(result.merges, result.vocab, specials, pattern)
        tok.train_stats = dict(result.stats)
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ----------------------------------------------------------------- encode
    def _encode_chunk(self, raw: bytes) -> list[int]:
        """Apply merges to a single pre-token, lowest merge index first.

        Order matters: applying the *most frequent applicable* merge instead of
        the *earliest learned* one produces a different (and wrong) tokenization
        that still decodes correctly, which makes it a nasty silent bug.
        """
        ids = list(raw)
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            candidate = min(
                (p for p in pairs if p in self.merges), key=lambda p: self.merges[p], default=None
            )
            if candidate is None:
                break
            new_id = self.merges[candidate]
            out: list[int] = []
            i = 0
            a, b = candidate
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode, treating any special-token text as ordinary characters."""
        out: list[int] = []
        for chunk in pretokenize(text, self.pattern):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def encode(self, text: str, allowed_special: SpecialPolicy = "none") -> list[int]:
        """Encode `text` to token ids.

        Args:
            allowed_special: `"none"` encodes special-token text literally
                (the safe default for untrusted input - otherwise a user can
                inject `<|endoftext|>` into a prompt); `"all"` maps it to its
                reserved id; `"raise"` refuses.

        Raises:
            ValueError: if `allowed_special="raise"` and a special token appears.
        """
        if not self.special_tokens or allowed_special == "none":
            return self.encode_ordinary(text)
        if allowed_special == "raise":
            for tok in self.special_tokens:
                if tok in text:
                    raise ValueError(f"special token {tok!r} found in input text")
            return self.encode_ordinary(text)

        pattern = "(" + "|".join(regex.escape(k) for k in self.special_tokens) + ")"
        out: list[int] = []
        for part in regex.split(pattern, text):
            if not part:
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    # ----------------------------------------------------------------- decode
    def decode(self, ids: list[int], errors: str = "replace") -> str:
        """Decode ids back to text.

        `errors="replace"` matters: a *prefix* of a multi-byte character is a
        valid token sequence, so streaming decoders hit invalid UTF-8 constantly.
        """
        parts: list[bytes] = []
        for i in ids:
            if i in self._inverse_special:
                parts.append(self._inverse_special[i].encode("utf-8"))
            elif i in self.vocab:
                parts.append(self.vocab[i])
            else:
                raise ValueError(f"unknown token id {i}")
        return b"".join(parts).decode("utf-8", errors=errors)

    # ------------------------------------------------------------------- i/o
    def save(self, path: str | Path) -> None:
        """Serialise to JSON. Merges are stored as an ordered list, since order
        is semantically load-bearing and JSON object keys are not ordered."""
        payload = {
            "version": 1,
            "pattern": self.pattern,
            "merges": [[a, b, new] for (a, b), new in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        payload = json.loads(Path(path).read_text())
        merges = {(a, b): new for a, b, new in payload["merges"]}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new in merges.items():  # rebuild vocab by replaying merges
            vocab[new] = vocab[a] + vocab[b]
        specials = {k: int(v) for k, v in payload.get("special_tokens", {}).items()}
        for tok, tid in specials.items():
            vocab[tid] = tok.encode("utf-8")
        return cls(merges, vocab, specials, payload.get("pattern"))

    # ------------------------------------------------------------- reporting
    def compression_ratio(self, text: str) -> float:
        """UTF-8 bytes per token. Higher is better; 1.0 means byte-level."""
        n = len(self.encode_ordinary(text))
        return len(text.encode("utf-8")) / max(n, 1)
