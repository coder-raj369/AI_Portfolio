from __future__ import annotations

from dataclasses import dataclass

from codecopilot.retriever import tokenize


@dataclass
class RankedDocument:
    path: str
    score: float
    snippet: str
    reasons: list[str]


class Reranker:
    """Simple reranker that combines BM25 score with symbol and title overlap."""

    def rerank(self, query: str, candidates: list[tuple[str, float]], *, top_k: int = 5) -> list[RankedDocument]:
        query_tokens = set(tokenize(query))
        ranked: list[RankedDocument] = []
        for path, score in candidates:
            text = path
            reasons: list[str] = []
            tokens = set(tokenize(path))
            overlap = len(query_tokens & tokens)
            symbol_bonus = 2.0 * overlap
            title_bonus = 1.0 if any(token in path.lower() for token in query_tokens) else 0.0

            if overlap:
                reasons.append(f"symbol overlap ({overlap})")
            if title_bonus:
                reasons.append("path title match")

            final_score = score + symbol_bonus + title_bonus
            ranked.append(RankedDocument(path=path, score=final_score, snippet=text, reasons=reasons))

        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]
