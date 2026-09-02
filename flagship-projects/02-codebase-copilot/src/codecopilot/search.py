from __future__ import annotations

from dataclasses import dataclass

from codecopilot.ranker import Reranker
from codecopilot.retriever import BM25Retriever


@dataclass
class Document:
    path: str
    title: str
    body: str


class CodebaseSearch:
    """Search a small set of codebase documents using lexical relevance plus reranking."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.retriever = BM25Retriever()
        self.reranker = Reranker()

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, object]]:
        corpus = [f"{doc.title}\n{doc.body}" for doc in self.documents]
        raw = self.retriever.search(query, corpus, top_k=max(10, top_k * 3))
        reranked = self.reranker.rerank(query, raw, top_k=top_k)

        results = []
        for item in reranked:
            source = next(doc for doc in self.documents if f"{doc.title}\n{doc.body}" == item.snippet)
            results.append({
                "path": source.path,
                "title": source.title,
                "score": item.score,
                "reasons": item.reasons,
            })
        return results
