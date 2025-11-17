# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class Dense(Module):
    def __init__(self: Dense,
                 in_dim: int,
                 out_dim: int) -> None:
        self.W: Parameter = Parameter(np.random.randn(in_dim, out_dim))
        self.b: Parameter = Parameter(np.random.randn(1, out_dim))

    # X has shape [num_examples, in_dim]
    def forward(self: Dense,
                X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.W.val) + self.b.val

    # because we have learnable parameters here,
    # we need to do 3 things:
    #   1) compute dLoss_dW
    #   2) compute dLoss_db
    #   3) compute (and return) dLoss_dX
    def backward(self: Dense,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        dModule_dW: np.ndarray = X.T
        dModule_dX: np.ndarray = self.W.val.T

        self.W.grad += np.dot(X.T, dLoss_dModule)
        self.b.grad += np.sum(dLoss_dModule, axis=0, keepdims=True)

        return np.dot(dLoss_dModule, dModule_dX)

    def parameters(self: Dense) -> List[Parameter]:
        return [self.W, self.b]

