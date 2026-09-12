from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

from codecopilot.search import CodebaseSearch


@dataclass(frozen=True)
class PatchProposal:
    """A validated, non-mutating patch preview."""

    path: str
    diff: str
    occurrences: int


class CodebaseTools:
    """Safe repository tools for search, bounded reads, and patch proposals."""

    def __init__(self, root: str | Path, search: CodebaseSearch | None = None) -> None:
        self.root = Path(root).resolve()
        self.search_index = search

    def _resolve_path(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("path must remain inside the repository root")
        return candidate

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, object]]:
        if self.search_index is None:
            raise RuntimeError("CodebaseTools.search requires a CodebaseSearch index")
        return self.search_index.search(query, top_k=top_k)

    def read_file(
        self,
        relative_path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, object]:
        """Read a bounded, 1-based inclusive range without leaving the root."""
        if start_line < 1 or (end_line is not None and end_line < start_line):
            raise ValueError("line range must be positive and ordered")
        path = self._resolve_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        lines = path.read_text(encoding="utf-8").splitlines()
        selected = lines[start_line - 1 : end_line]
        return {
            "path": relative_path,
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1,
            "content": "\n".join(selected),
        }

    def propose_patch(self, relative_path: str, old_text: str, new_text: str) -> PatchProposal:
        """Validate one exact replacement and return a unified diff without writing it."""
        path = self._resolve_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        original = path.read_text(encoding="utf-8")
        occurrences = original.count(old_text)
        if occurrences != 1:
            raise ValueError(f"expected exactly one replacement target, found {occurrences}")
        updated = original.replace(old_text, new_text, 1)
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=relative_path,
            tofile=relative_path,
        ))
        return PatchProposal(path=relative_path, diff=diff, occurrences=occurrences)
