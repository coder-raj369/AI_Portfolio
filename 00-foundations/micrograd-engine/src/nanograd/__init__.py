"""nanograd: a small, broadcasting-correct reverse-mode autodiff engine in NumPy.

The point of this package is to make the mechanics of backpropagation explicit:
every operation records a local gradient closure, and `Tensor.backward()` walks
the graph in reverse topological order. Correctness is not asserted, it is
verified against central finite differences in `tests/test_gradcheck.py`.
"""

from nanograd.data import make_spirals, train_test_split
from nanograd.gradcheck import gradcheck, numeric_grad
from nanograd.nn import MLP, Linear, Module, ReLU, Sequential, Tanh
from nanograd.tensor import Tensor
from nanograd.functional import (
    cross_entropy,
    log_softmax,
    mse_loss,
    nll_loss,
    relu,
    sigmoid,
    softmax,
    tanh,
)

__all__ = [
    "MLP",
    "Linear",
    "Module",
    "ReLU",
    "Sequential",
    "Tanh",
    "Tensor",
    "cross_entropy",
    "gradcheck",
    "log_softmax",
    "make_spirals",
    "mse_loss",
    "nll_loss",
    "numeric_grad",
    "relu",
    "sigmoid",
    "softmax",
    "tanh",
    "train_test_split",
]
__version__ = "0.1.0"
