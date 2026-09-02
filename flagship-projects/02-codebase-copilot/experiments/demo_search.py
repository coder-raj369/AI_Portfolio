from __future__ import annotations

from codecopilot.search import CodebaseSearch, Document


DOCS = [
    Document(path="src/tokenizer.py", title="Tokenizer training", body="Byte-level BPE trainer with merge table, vocab sizing, and encode/decode routines for text and special tokens."),
    Document(path="src/train.py", title="Training loop", body="Pretraining loop for a small decoder model with cross-entropy loss, gradient clipping, and checkpointing."),
    Document(path="src/retriever.py", title="Retrieval index", body="BM25 lexical search over file content and titles to rank candidate documents for codebase questions."),
    Document(path="src/eval.py", title="Evaluation harness", body="Pass@k scoring, exact match, numeric match, and grading utilities for benchmark runs."),
    Document(path="src/ranker.py", title="Reranking logic", body="Refines BM25 retrieval with filename and symbol match bonuses to prioritize likely relevant files."),
]


if __name__ == "__main__":
    search = CodebaseSearch(DOCS)
    for result in search.search("how do we train the tokenizer and rank code files", top_k=3):
        print(result)
