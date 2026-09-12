from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeChunk:
    """A syntax-aware source chunk with a stable qualified symbol name."""

    path: str
    symbol: str
    start_line: int
    end_line: int
    source: str


def chunk_python_source(source: str, *, path: str = "<memory>") -> list[CodeChunk]:
    """Extract classes and functions from Python source using the standard AST."""
    tree = ast.parse(source, filename=path)
    chunks: list[CodeChunk] = []
    lines = source.splitlines()

    def visit(nodes: list[ast.AST], parents: tuple[str, ...] = ()) -> None:
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = ".".join((*parents, node.name))
                end_line = getattr(node, "end_lineno", node.lineno)
                chunks.append(CodeChunk(
                    path=path,
                    symbol=symbol,
                    start_line=node.lineno,
                    end_line=end_line,
                    source="\n".join(lines[node.lineno - 1 : end_line]),
                ))
                if isinstance(node, ast.ClassDef):
                    visit(node.body, (*parents, node.name))

    visit(tree.body)
    return chunks
