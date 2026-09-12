from __future__ import annotations

import math
import re
from collections import Counter
from hashlib import sha256

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


class HybridRetriever:
    """Fuse BM25 with a deterministic dense-style feature similarity baseline."""

    def __init__(self, *, lexical_weight: float = 0.7, dimensions: int = 128) -> None:
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between 0 and 1")
        if dimensions < 16:
            raise ValueError("dimensions must be at least 16")
        self.lexical_weight = lexical_weight
        self.dimensions = dimensions
        self.bm25 = BM25Retriever()

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        features = tokens + [
            token[i : i + 3]
            for token in tokens
            for i in range(max(1, len(token) - 2))
        ]
        for feature in features:
            digest = sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def search(self, query: str, documents: list[str], *, top_k: int = 5) -> list[tuple[str, float]]:
        if not documents:
            return []
        query_vector = self._vectorize(query)
        doc_vectors = [self._vectorize(doc) for doc in documents]
        dense_scores = [
            sum(query_value * doc_value for query_value, doc_value in zip(query_vector, vector))
            for vector in doc_vectors
        ]
        lexical = self.bm25.search(query, documents, top_k=len(documents))
        lexical_scores = {doc: score for doc, score in lexical}
        max_lexical = max(lexical_scores.values(), default=0.0)
        max_dense = max(dense_scores, default=0.0)
        dense_weight = 1.0 - self.lexical_weight
        scored = []
        for doc, dense_score in zip(documents, dense_scores):
            lexical_score = lexical_scores.get(doc, 0.0)
            normalized_lexical = lexical_score / max_lexical if max_lexical else 0.0
            normalized_dense = dense_score / max_dense if max_dense > 0.0 else 0.0
            scored.append((doc, self.lexical_weight * normalized_lexical + dense_weight * normalized_dense))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
