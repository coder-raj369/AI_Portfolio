"""bpetok: byte-level BPE, trained and applied from scratch.

Byte-level means the base vocabulary is the 256 possible bytes, so the tokenizer
is lossless on *any* input - emoji, CJK, control characters, invalid UTF-8
fragments - without an `<unk>` token. That property is asserted directly in
`tests/test_roundtrip.py`.
"""

from bpetok.tokenizer import BPETokenizer
from bpetok.regex_split import GPT4_SPLIT_PATTERN, pretokenize
from bpetok.trainer import TrainResult, train_bpe

__all__ = [
    "GPT4_SPLIT_PATTERN",
    "BPETokenizer",
    "TrainResult",
    "pretokenize",
    "train_bpe",
]
__version__ = "0.1.0"
