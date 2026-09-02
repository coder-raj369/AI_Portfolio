from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Retriever:
    """A compact BM25-like lexical scorer for code and docs."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def search(self, query: str, documents: list[str], *, top_k: int = 5) -> list[tuple[str, float]]:
        if not documents:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return [(doc, 0.0) for doc in documents[:top_k]]

        doc_tokens = [tokenize(doc) for doc in documents]
        doc_freq = {}
        for tokens in doc_tokens:
            seen = set()
            for token in tokens:
                if token not in seen:
                    doc_freq[token] = doc_freq.get(token, 0) + 1
                    seen.add(token)

        avgdl = sum(len(tokens) for tokens in doc_tokens) / len(doc_tokens)
        scored: list[tuple[str, float]] = []
        n_docs = len(documents)

        for doc, tokens in zip(documents, doc_tokens):
            score = 0.0
            token_counts = Counter(tokens)
            doc_len = len(tokens)
            for token in set(query_tokens):
                if token not in token_counts:
                    continue
                tf = token_counts[token]
                df = doc_freq.get(token, 0)
                idf = math.log(((n_docs - df + 0.5) / (df + 0.5)) + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
                score += idf * ((self.k1 + 1.0) * tf) / denominator
            scored.append((doc, score))

        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
