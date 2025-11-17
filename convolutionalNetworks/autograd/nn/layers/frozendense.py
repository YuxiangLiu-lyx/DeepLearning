# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


"""
    In case you aren't comfortable with your Dense module (or you didn't get it to work) from pa1,
    this module implements a Dense layer where the params are frozen (e.g. cannot be changed when training).
    Most CNNs have some Dense layers in there at some point, so feel free to use this layer instead of your
    Dense layer if you're not confident. Of course, if your Dense layer works, then use it instead of this!
"""
class FrozenDense(Module):
    def __init__(self: FrozenDense,
                 in_dim: int,
                 out_dim: int) -> None:
        self.W: Parameter = Parameter(np.random.randn(in_dim, out_dim))
        self.b: Parameter = Parameter(np.random.randn(1, out_dim))

        # WARNING: if you unfreeze these params, this module will no longer be correct!
        self.W.freeze()
        self.b.freeze()

    # X has shape [num_examples, in_dim]
    def forward(self: FrozenDense,
                X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.W.val) + self.b.val

    # because our params are frozen we only need to compute and return dLoss_dX
    def backward(self: FrozenDense,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        dModule_dX: np.ndarray = self.W.val.T
        return np.dot(dLoss_dModule, dModule_dX)

    def parameters(self: FrozenDense) -> List[Parameter]:
        return [self.W, self.b]

