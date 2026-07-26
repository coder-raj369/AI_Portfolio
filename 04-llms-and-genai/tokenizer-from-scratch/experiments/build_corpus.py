"""Assemble a training corpus with no network access and no licensing questions.

Default source: **this repository's own text** (Markdown prose + Python source +
config), which is MIT-licensed by us and therefore safe to redistribute. It is a
mixed prose/code register, which is realistic for a code-assistant tokenizer but
*not* representative of natural-language-only text - the README says so where it
quotes compression numbers.

For natural-language numbers, pass any UTF-8 text file:

    python experiments/train_and_benchmark.py --input path/to/fineweb_sample.txt

(FineWeb-Edu is ODC-By; download it separately with `datasets` if you want it.)
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SUFFIXES = (".md", ".py", ".toml", ".yml", ".cfg")
SKIP_PARTS = {".git", "__pycache__", "results", ".venv", ".ruff_cache", ".pytest_cache"}


def build_repo_corpus(root: Path | None = None) -> str:
    """Concatenate every text file in the repo, in a stable (sorted) order."""
    root = root or REPO_ROOT
    chunks: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        if SKIP_PARTS & set(path.parts):
            continue
        chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(chunks)


def split(corpus: str, train_frac: float = 0.85) -> tuple[str, str]:
    """Split into train / held-out. Compression measured on the training text
    overstates quality, so both numbers get reported."""
    cut = int(len(corpus) * train_frac)
    return corpus[:cut], corpus[cut:]


if __name__ == "__main__":
    text = build_repo_corpus()
    print(f"{len(text):,} characters, {len(text.encode('utf-8')):,} bytes")
