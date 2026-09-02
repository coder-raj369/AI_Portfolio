from __future__ import annotations

from typing import Iterable

from bpetok import BPETokenizer


def build_pretrain_corpus() -> str:
    """A compact synthetic corpus with simple factual + relational patterns."""
    names = ["Ava", "Leo", "Nia", "Iris", "Omar", "Zoe", "Kai", "Mila"]
    places = ["Paris", "Tokyo", "Rome", "Lisbon", "Seoul", "Berlin", "Cairo", "Oslo"]
    animals = ["fox", "owl", "falcon", "otter", "leopard", "dolphin", "badger", "yak"]
    verbs = ["observes", "visits", "studies", "tracks", "follows", "respects", "reports", "scores"]
    adjectives = ["bright", "calm", "quiet", "curious", "swift", "gentle", "lively", "precise"]

    rows: list[str] = []
    for i in range(240):
        name = names[i % len(names)]
        place = places[(i * 3) % len(places)]
        animal = animals[(i * 5) % len(animals)]
        verb = verbs[i % len(verbs)]
        adj = adjectives[(i * 2) % len(adjectives)]
        rows.append(f"{name} {verb} the {adj} {animal} near {place}.")
    rows.append("The capital of France is Paris.")
    rows.append("Two plus two equals four.")
    rows.append("A red fox is quick and alert.")
    rows.append("The sky is blue in clear weather.")
    return "\n".join(rows)


def build_instruction_pairs() -> list[tuple[str, str]]:
    """Small instruction-tuning set that can be learned quickly on CPU."""
    return [
        ("What is the capital of France?", "Paris."),
        ("What is 3 + 4?", "7."),
        ("Name a color often seen at sunrise.", "Orange."),
        ("What animal is known for being clever?", "Fox."),
        ("Which city is known for the Eiffel Tower?", "Paris."),
        ("What is 9 - 2?", "7."),
        ("Say the word for a large body of water.", "Ocean."),
        ("What do we call a very fast bird?", "Falcon."),
        ("What planet do we live on?", "Earth."),
        ("If a box has 5 apples and you add 3, how many?", "8."),
    ]


def build_preference_pairs() -> list[tuple[str, str, str]]:
    """Simple chosen/rejected answer pairs for DPO."""
    return [
        ("What is the capital of Italy?", "Rome.", "Milan."),
        ("What is 6 + 6?", "12.", "10."),
        ("Which planet is known as the Red Planet?", "Mars.", "Venus."),
        ("What animal is a common farm animal?", "Cow.", "Horse."),
        ("What color is grass usually?", "Green.", "Blue."),
        ("What is 5 * 3?", "15.", "12."),
    ]


def train_tokenizer(corpus: str, vocab_size: int = 512) -> BPETokenizer:
    if vocab_size < 258:
        raise ValueError("vocab_size must be at least 258 to reserve the byte alphabet and special tokens")
    return BPETokenizer.train(corpus, vocab_size=vocab_size, special_tokens=["<|pad|>", "<|endoftext|>"])


def encode_text(tokenizer: BPETokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, allowed_special="all")


def make_training_sequences(tokenizer: BPETokenizer, max_len: int = 32) -> list[list[int]]:
    corpus = build_pretrain_corpus()
    tokens = encode_text(tokenizer, corpus)
    return [tokens[i : i + max_len] for i in range(0, len(tokens), max_len)]


def make_instruction_sequences(tokenizer: BPETokenizer) -> list[tuple[list[int], list[int]]]:
    inputs: list[tuple[list[int], list[int]]] = []
    for prompt, answer in build_instruction_pairs():
        prompt_ids = encode_text(tokenizer, prompt)
        answer_ids = encode_text(tokenizer, answer)
        inputs.append((prompt_ids, answer_ids))
    return inputs


def make_preference_sequences(tokenizer: BPETokenizer) -> list[tuple[list[int], list[int], list[int]]]:
    pairs: list[tuple[list[int], list[int], list[int]]] = []
    for prompt, chosen, rejected in build_preference_pairs():
        pairs.append((encode_text(tokenizer, prompt), encode_text(tokenizer, chosen), encode_text(tokenizer, rejected)))
    return pairs


def iter_batches(seq_list: Iterable[list[int]], batch_size: int) -> list[list[list[int]]]:
    items = list(seq_list)
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
