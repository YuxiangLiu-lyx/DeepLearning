# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class Flatten(Module):
    X = None
    original_shape = None

    def forward(self: Flatten,
                X: np.ndarray) -> np.ndarray:
        self.X = X
        self.original_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self: Flatten,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        
        if self.original_shape is None:
            return dLoss_dModule.reshape(X.shape)
        return dLoss_dModule.reshape(self.original_shape)

    def parameters(self: Flatten) -> List[Parameter]:
        return list()

