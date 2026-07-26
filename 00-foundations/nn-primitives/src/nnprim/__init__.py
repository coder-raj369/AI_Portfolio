"""nnprim: the numerical guts of a transformer block, written out in NumPy.

Three things live here, all of which a framework normally hides:

1. **Numerically stable primitives** - softmax, logsumexp, cross-entropy. The
   naive formulas are correct in exact arithmetic and produce `nan` in float32
   above a logit of ~89. `experiments/stability_report.py` measures exactly
   where each one breaks.
2. **Normalisation with hand-derived backward passes** - LayerNorm and RMSNorm
   forward *and* backward, verified against finite differences.
3. **Attention from raw tensor operations** - no `nn.MultiheadAttention`,
   including a causal-masking test that perturbs the future and asserts the
   present does not move.
"""

from nnprim.attention import (
    causal_mask,
    multi_head_attention,
    naive_multi_head_attention,
    scaled_dot_product_attention,
    split_heads,
)
from nnprim.gradcheck import max_abs_error, numeric_grad
from nnprim.functional import (
    cross_entropy_from_logits,
    gelu,
    log_softmax,
    logsumexp,
    sigmoid,
    silu,
    softmax,
    swiglu,
)
from nnprim.init import (
    kaiming_normal,
    kaiming_uniform,
    residual_scaled_normal,
    truncated_normal,
    xavier_normal,
    xavier_uniform,
)
from nnprim.norm import layer_norm, layer_norm_backward, rms_norm, rms_norm_backward

__all__ = [
    "causal_mask",
    "cross_entropy_from_logits",
    "gelu",
    "kaiming_normal",
    "kaiming_uniform",
    "layer_norm",
    "layer_norm_backward",
    "log_softmax",
    "logsumexp",
    "max_abs_error",
    "multi_head_attention",
    "naive_multi_head_attention",
    "numeric_grad",
    "residual_scaled_normal",
    "rms_norm",
    "rms_norm_backward",
    "scaled_dot_product_attention",
    "sigmoid",
    "silu",
    "softmax",
    "split_heads",
    "swiglu",
    "truncated_normal",
    "xavier_normal",
    "xavier_uniform",
]
__version__ = "0.1.0"
