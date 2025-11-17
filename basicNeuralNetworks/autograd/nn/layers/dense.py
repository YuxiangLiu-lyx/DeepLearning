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
                 out_dim: int):
        w_data = np.random.randn(in_dim, out_dim)
        b_data = np.random.randn(1, out_dim)

        self.W = Parameter(w_data)
        self.b = Parameter(b_data)

        self.X = None

    def forward(self: Dense,
                X: np.ndarray) -> np.ndarray:
        self.X = X
        return self.X @ self.W.val + self.b.val
    
    def backward(self: Dense,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        if self.X is None:
            self.X = X
        if not self.W.frozen:
            if self.W.grad is None:
                self.W.grad = self.X.T @ dLoss_dModule
            else:
                self.W.grad += self.X.T @ dLoss_dModule

        if not self.b.frozen:
            if self.b.grad is None:
                self.b.grad = np.sum(dLoss_dModule, axis=0, keepdims=True)
            else:
                self.b.grad += np.sum(dLoss_dModule, axis=0, keepdims=True)
        return dLoss_dModule @ self.W.val.T


    def parameters(self: Dense) -> List[Parameter]:
        return [self.W, self.b]

