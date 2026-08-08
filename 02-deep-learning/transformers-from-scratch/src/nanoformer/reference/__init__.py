"""Executable NumPy reference implementations.

These are the ground truth for the PyTorch implementations in
`nanoformer.torch_impl`: they run anywhere NumPy runs (no GPU, no downloads), and
the torch tests assert numerical agreement against them. Building the reference
first is what makes the PyTorch version *verifiable* rather than merely plausible.
"""

from nanoformer.reference.attention import (
    GroupedQueryAttention,
    KVCache,
    repeat_kv,
    softmax,
)
from nanoformer.reference.moe import (
    MoEFeedForward,
    expert_utilisation,
    load_balancing_loss,
    router_z_loss,
    top_k_router,
)
from nanoformer.reference.positions import (
    alibi_bias,
    alibi_slopes,
    apply_rope,
    rope_frequencies,
)
from nanoformer.reference.ssm import (
    SelectiveSSMBlock,
    lti_scan_via_convolution,
    selective_scan_chunked,
    selective_scan_sequential,
    softplus,
)

__all__ = [
    "GroupedQueryAttention",
    "KVCache",
    "MoEFeedForward",
    "SelectiveSSMBlock",
    "alibi_bias",
    "alibi_slopes",
    "apply_rope",
    "expert_utilisation",
    "load_balancing_loss",
    "lti_scan_via_convolution",
    "repeat_kv",
    "rope_frequencies",
    "router_z_loss",
    "selective_scan_chunked",
    "selective_scan_sequential",
    "softmax",
    "softplus",
    "top_k_router",
]
