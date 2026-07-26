"""BPE training: iteratively merge the most frequent adjacent pair.

The naive implementation recounts every pair on every merge, which is
O(merges x corpus). This one keeps an inverted index from pair -> the
pre-token chunks containing it, so a merge only touches the chunks it affects.
On the corpora used here that is the difference between ~3 minutes and ~2 seconds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from bpetok.regex_split import pretokenize


@dataclass
class TrainResult:
    """Output of `train_bpe`.

    Attributes:
        merges: ordered `(pair) -> new_id` map. Order *is* the algorithm: encoding
            must apply merges in exactly this sequence.
        vocab: `id -> bytes` for every token, including the 256 base bytes.
        stats: diagnostics recorded during training.
    """

    merges: dict[tuple[int, int], int]
    vocab: dict[int, bytes]
    stats: dict[str, float | int] = field(default_factory=dict)


def _pair_counts(words: list[list[int]], freqs: list[int]):
    counts: Counter[tuple[int, int]] = Counter()
    index: dict[tuple[int, int], set[int]] = {}
    for wi, word in enumerate(words):
        f = freqs[wi]
        for pair in zip(word, word[1:]):
            counts[pair] += f
            index.setdefault(pair, set()).add(wi)
    return counts, index


def _merge_word(word: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    a, b = pair
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def train_bpe(
    text: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    pattern: str | None = None,
    verbose: bool = False,
) -> TrainResult:
    """Train a byte-level BPE vocabulary.

    Args:
        text: training corpus.
        vocab_size: total target size *including* the 256 byte tokens and any
            special tokens.
        special_tokens: strings that get reserved ids at the end of the vocab and
            are never produced by merges.
        pattern: override the pre-tokenization regex.

    Returns:
        A `TrainResult`.

    Raises:
        ValueError: if `vocab_size` leaves no room for merges.
    """
    special_tokens = special_tokens or []
    n_special = len(special_tokens)
    n_merges = vocab_size - 256 - n_special
    if n_merges < 0:
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than 256 bytes + {n_special} special tokens"
        )

    chunk_freqs = Counter(pretokenize(text, pattern))
    words = [list(chunk.encode("utf-8")) for chunk in chunk_freqs]
    freqs = list(chunk_freqs.values())

    counts, index = _pair_counts(words, freqs)
    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    for step in range(n_merges):
        if not counts:
            break
        # Ties are broken by the pair itself, so training is deterministic
        # regardless of dict iteration order.
        best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
        if counts[best] < 2:
            break
        new_id = 256 + step
        merges[best] = new_id
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]

        for wi in list(index.get(best, ())):
            word = words[wi]
            if len(word) < 2:
                continue
            f = freqs[wi]
            for pair in zip(word, word[1:]):  # retract this chunk's old pairs
                counts[pair] -= f
                if counts[pair] <= 0:
                    counts.pop(pair, None)
                    index.pop(pair, None)
                elif wi in index.get(pair, ()):
                    index[pair].discard(wi)
            new_word = _merge_word(word, best, new_id)
            words[wi] = new_word
            for pair in zip(new_word, new_word[1:]):  # and add the new ones
                counts[pair] += f
                index.setdefault(pair, set()).add(wi)
        counts.pop(best, None)
        index.pop(best, None)
        if verbose and (step + 1) % 200 == 0:
            print(f"  merge {step + 1}/{n_merges}: {vocab[new_id]!r}")

    for i, token in enumerate(special_tokens):
        vocab[256 + len(merges) + i] = token.encode("utf-8")

    total_bytes = len(text.encode("utf-8"))
    stats = {
        "corpus_bytes": total_bytes,
        "unique_pretokens": len(chunk_freqs),
        "merges_learned": len(merges),
        "vocab_size": len(vocab),
    }
    return TrainResult(merges=merges, vocab=vocab, stats=stats)
