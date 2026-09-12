"""Lightweight codebase search and retrieval package for a portfolio flagship."""

from codecopilot.ast_chunker import CodeChunk, chunk_python_source
from codecopilot.ranker import Reranker
from codecopilot.retriever import BM25Retriever, HybridRetriever, tokenize
from codecopilot.search import CodebaseSearch, Document
from codecopilot.tools import CodebaseTools, PatchProposal

__all__ = [
    "BM25Retriever",
    "CodeChunk",
    "CodebaseSearch",
    "CodebaseTools",
    "Document",
    "HybridRetriever",
    "PatchProposal",
    "Reranker",
    "chunk_python_source",
    "tokenize",
]

__version__ = "0.1.0"
