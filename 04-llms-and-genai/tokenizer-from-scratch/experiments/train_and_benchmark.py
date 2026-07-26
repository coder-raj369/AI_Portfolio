"""Train BPE at several vocabulary sizes and report compression, speed and examples.

Every number in this module's README comes from here.

Run:
    python experiments/train_and_benchmark.py
    python experiments/train_and_benchmark.py --input my_corpus.txt --sizes 512 1024 4096
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bpetok import BPETokenizer  # noqa: E402

from build_corpus import build_repo_corpus, split  # noqa: E402  (local module)

RESULTS = Path(__file__).parent / "results"
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
SAMPLE = (
    "def compute_attention(q, k, v):\n"
    "    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])\n"
    "    return softmax(scores, axis=-1) @ v\n"
)


def run(corpus: str, sizes: list[int]) -> dict:
    train_text, held_out = split(corpus)
    rows = []
    byte_baseline = 1.0
    for size in sizes:
        start = time.perf_counter()
        tok = BPETokenizer.train(train_text, vocab_size=size, special_tokens=SPECIALS)
        train_seconds = time.perf_counter() - start

        enc_start = time.perf_counter()
        ids = tok.encode_ordinary(held_out)
        enc_seconds = time.perf_counter() - enc_start

        assert tok.decode(ids) == held_out, "held-out roundtrip must be lossless"
        rows.append(
            {
                "requested_vocab": size,
                "actual_vocab": tok.vocab_size,
                "merges": len(tok.merges),
                "train_seconds": round(train_seconds, 2),
                "bytes_per_token_train": round(tok.compression_ratio(train_text), 3),
                "bytes_per_token_heldout": round(tok.compression_ratio(held_out), 3),
                "vs_byte_level": round(tok.compression_ratio(held_out) / byte_baseline, 3),
                "encode_mb_per_second": round(
                    len(held_out.encode("utf-8")) / 1e6 / max(enc_seconds, 1e-9), 3
                ),
                "sample_tokens": [tok.decode([i]) for i in tok.encode_ordinary(SAMPLE)][:24],
                "longest_tokens": sorted(
                    (tok.vocab[i].decode("utf-8", errors="replace") for i in
                     range(256, 256 + len(tok.merges))),
                    key=len, reverse=True,
                )[:12],
            }
        )
        print(f"vocab {size:>5} -> {rows[-1]['actual_vocab']:>5} tokens | "
              f"held-out {rows[-1]['bytes_per_token_heldout']:.3f} B/tok | "
              f"train {rows[-1]['train_seconds']:.2f}s | "
              f"encode {rows[-1]['encode_mb_per_second']:.3f} MB/s")

    best = max(rows, key=lambda r: r["bytes_per_token_heldout"])
    tok = BPETokenizer.train(train_text, vocab_size=best["requested_vocab"],
                             special_tokens=SPECIALS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tok.save(RESULTS / "tokenizer.json")

    payload = {
        "corpus": {
            "source": "this repository's own .md/.py/.toml/.yml text (MIT)",
            "total_bytes": len(corpus.encode("utf-8")),
            "train_bytes": len(train_text.encode("utf-8")),
            "heldout_bytes": len(held_out.encode("utf-8")),
        },
        "note": "mixed prose+code register; natural-language-only corpora "
                "typically compress ~10-20% better at the same vocab size",
        "rows": rows,
    }
    (RESULTS / "benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot(rows)
    return payload


def plot(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=140)
    sizes = [r["actual_vocab"] for r in rows]
    ax.plot(sizes, [r["bytes_per_token_heldout"] for r in rows], "o-", lw=1.7, label="held-out")
    ax.plot(sizes, [r["bytes_per_token_train"] for r in rows], "s--", lw=1.3, label="training text")
    ax.axhline(1.0, ls=":", c="grey", lw=1)
    ax.text(sizes[0], 1.05, "byte-level baseline", fontsize=7, color="grey")
    ax.set(xlabel="vocabulary size", ylabel="bytes per token (higher = better)",
           title="BPE compression vs vocabulary size")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "compression.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="UTF-8 text file to train on (default: this repo's text)")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192])
    args = parser.parse_args()
    corpus = args.input.read_text(encoding="utf-8") if args.input else build_repo_corpus()
    print(f"corpus: {len(corpus.encode('utf-8')):,} bytes")
    run(corpus, args.sizes)


if __name__ == "__main__":
    main()
