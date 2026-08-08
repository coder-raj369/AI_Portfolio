"""nanoformer: transformer, MoE and SSM blocks built from raw tensor operations.

Two parallel implementations of the same algorithms:

- `nanoformer.reference` - NumPy, dependency-light, runs anywhere. This is the
  ground truth.
- `nanoformer.torch_impl` - PyTorch, the version you would actually train with,
  tested for numerical agreement against the reference.

Nothing here calls `nn.MultiheadAttention`, `nn.RMSNorm` or a fused SSM kernel;
the point is the arithmetic, and the equivalence tests are what make it credible.
"""

__version__ = "0.1.0"
