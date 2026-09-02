from __future__ import annotations

from codecopilot.ranker import Reranker
from codecopilot.retriever import BM25Retriever
from codecopilot.search import CodebaseSearch, Document


def test_bm25_retriever_prioritizes_query_terms() -> None:
    docs = [
        "train the tokenizer with merge operations and byte vocabulary",
        "serve a web api that returns health checks",
        "rerank candidate files by path and symbol overlap",
    ]
    ranked = BM25Retriever().search("tokenizer train merge vocabulary", docs, top_k=2)
    assert ranked[0][0] == docs[0]


def test_reranker_keeps_best_matches_first() -> None:
    reranker = Reranker()
    ranked = reranker.rerank("tokenizer merge train", [("src/train.py", 1.0), ("src/tokenizer.py", 3.0), ("src/server.py", 0.2)], top_k=2)
    assert ranked[0].path == "src/tokenizer.py"
    assert len(ranked) == 2


def test_codebase_search_returns_paths_and_scores() -> None:
    docs = [
        Document(path="src/tokenizer.py", title="Tokenizer training", body="Byte-level BPE trainer and encoder"),
        Document(path="src/train.py", title="Training loop", body="Cross entropy training loop for the decoder model"),
        Document(path="src/eval.py", title="Evaluation harness", body="Pass@k and verifier scoring"),
    ]
    search = CodebaseSearch(docs)
    results = search.search("how do we train the tokenizer", top_k=2)
    assert results[0]["path"] == "src/tokenizer.py"
    assert results[0]["score"] >= results[1]["score"]
