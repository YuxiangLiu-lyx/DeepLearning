# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class ReLU(Module):

    def forward(self: ReLU,
                X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    # also element-wise indepenent
    def backward(self: ReLU,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        gradient = X > 0
        return dLoss_dModule * gradient

    def parameters(self: ReLU) -> List[Parameter]:
        return list()

