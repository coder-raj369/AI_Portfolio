"""PyTorch implementations, tested against the NumPy reference.

Imports are lazy-friendly: this package requires `torch`, while
`nanoformer.reference` does not, so a torch-free environment can still use the
reference implementations and run most of the test suite.
"""

from nanoformer.torch_impl.attention import GroupedQueryAttention, KVCache, repeat_kv
from nanoformer.torch_impl.block import (
    Block,
    ModelConfig,
    SwiGLUFeedForward,
    TinyDecoderLM,
)
from nanoformer.torch_impl.moe import MoEFeedForward, MoEStats, SwiGLUExpert
from nanoformer.torch_impl.norm import RMSNorm
from nanoformer.torch_impl.positions import RotaryEmbedding, alibi_bias, alibi_slopes
from nanoformer.torch_impl.ssm import (
    SelectiveSSM,
    selective_scan_chunked,
    selective_scan_sequential,
)

__all__ = [
    "Block",
    "GroupedQueryAttention",
    "KVCache",
    "MoEFeedForward",
    "MoEStats",
    "ModelConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "SelectiveSSM",
    "SwiGLUExpert",
    "SwiGLUFeedForward",
    "TinyDecoderLM",
    "alibi_bias",
    "alibi_slopes",
    "repeat_kv",
    "selective_scan_chunked",
    "selective_scan_sequential",
]
