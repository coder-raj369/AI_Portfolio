from __future__ import annotations

from codecopilot.ast_chunker import chunk_python_source
from codecopilot.ranker import Reranker
from codecopilot.retriever import BM25Retriever
from codecopilot.search import CodebaseSearch, Document
from codecopilot.tools import CodebaseTools


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


def test_ast_chunker_preserves_qualified_symbols_and_spans() -> None:
    source = """class Tokenizer:\n    def encode(self, text):\n        return text\n\ndef train(model):\n    return model\n"""
    chunks = chunk_python_source(source, path="src/tokenizer.py")
    assert [(chunk.symbol, chunk.start_line) for chunk in chunks] == [
        ("Tokenizer", 1),
        ("Tokenizer.encode", 2),
        ("train", 5),
    ]
    assert "return text" in chunks[1].source


def test_symbol_aware_document_search() -> None:
    tokenizer = Document.from_python_source(
        "src/tokenizer.py",
        "class Tokenizer:\n    def encode(self, text):\n        return text\n",
    )
    server = Document.from_python_source(
        "src/server.py",
        "def health_check():\n    return True\n",
    )
    results = CodebaseSearch([tokenizer, server]).search("Tokenizer encode", top_k=1)
    assert results[0]["path"] == "src/tokenizer.py"
    assert results[0]["symbols"] == ["Tokenizer", "Tokenizer.encode"]


def test_tools_read_bounded_file_and_propose_non_mutating_patch(tmp_path) -> None:
    target = tmp_path / "module.py"
    target.write_text("def train():\n    return 1\n", encoding="utf-8")
    tools = CodebaseTools(tmp_path)

    result = tools.read_file("module.py", start_line=1, end_line=1)
    assert result["content"] == "def train():"

    proposal = tools.propose_patch("module.py", "return 1", "return 2")
    assert "-    return 1" in proposal.diff
    assert "+    return 2" in proposal.diff
    assert target.read_text(encoding="utf-8") == "def train():\n    return 1\n"


def test_tools_reject_traversal_and_ambiguous_patch(tmp_path) -> None:
    target = tmp_path / "module.py"
    target.write_text("x = 1\nx = 1\n", encoding="utf-8")
    tools = CodebaseTools(tmp_path)

    import pytest

    with pytest.raises(ValueError, match="inside"):
        tools.read_file("../module.py")
    with pytest.raises(ValueError, match="exactly one"):
        tools.propose_patch("module.py", "x = 1", "x = 2")
