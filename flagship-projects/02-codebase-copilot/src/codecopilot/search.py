from __future__ import annotations

from dataclasses import dataclass

from codecopilot.ast_chunker import chunk_python_source
from codecopilot.ranker import Reranker
from codecopilot.retriever import HybridRetriever


@dataclass
class Document:
    path: str
    title: str
    body: str
    symbols: tuple[str, ...] = ()

    @classmethod
    def from_python_source(cls, path: str, source: str) -> "Document":
        """Build a searchable document with AST-derived symbols."""
        chunks = chunk_python_source(source, path=path)
        return cls(
            path=path,
            title=path.rsplit("/", 1)[-1],
            body=source,
            symbols=tuple(chunk.symbol for chunk in chunks),
        )


class CodebaseSearch:
    """Search a small set of codebase documents using lexical relevance plus reranking."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.retriever = HybridRetriever()
        self.reranker = Reranker()

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, object]]:
        corpus = [
            f"{doc.title}\n{' '.join(doc.symbols)}\n{doc.body}"
            for doc in self.documents
        ]
        raw = self.retriever.search(query, corpus, top_k=max(10, top_k * 3))
        reranked = self.reranker.rerank(query, raw, top_k=top_k)

        results = []
        for item in reranked:
            source = self.documents[corpus.index(item.snippet)]
            results.append({
                "path": source.path,
                "title": source.title,
                "score": item.score,
                "reasons": item.reasons,
                "symbols": list(source.symbols),
            })
        return results
