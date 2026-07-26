"""nanoptim: optimisers and learning-rate schedules implemented from scratch.

Every optimiser here operates on any object exposing `.data` (an ndarray) and
`.grad` (an ndarray or `None`), which means the same code drives both the
`nanograd` autodiff engine from the sibling module and the plain-NumPy
`Parameter` used in the convex benchmarks.
"""

from nanoptim.optimizers import (
    SGD,
    Adam,
    AdamW,
    Lion,
    Optimizer,
    Parameter,
    clip_grad_norm_,
)
from nanoptim.schedules import (
    ConstantLR,
    CosineWithWarmup,
    Schedule,
    WarmupStableDecay,
)

__all__ = [
    "SGD",
    "Adam",
    "AdamW",
    "ConstantLR",
    "CosineWithWarmup",
    "Lion",
    "Optimizer",
    "Parameter",
    "Schedule",
    "WarmupStableDecay",
    "clip_grad_norm_",
]
__version__ = "0.1.0"
