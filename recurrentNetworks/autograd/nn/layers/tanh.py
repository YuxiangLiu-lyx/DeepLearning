# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class Tanh(Module):

    def forward(self: Tanh,
                X: np.ndarray) -> np.ndarray:
        return np.tanh(X)

    # also element-wise indepenent
    # derivative of tanh: (1- tanh(x)**2)
    def backward(self: Tanh,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        Y_hat: np.ndarray = self.forward(X)
        dModule_dX: np.ndarray = 1 - Y_hat**2
        return dLoss_dModule * dModule_dX

    def parameters(self: Tanh) -> List[Parameter]:
        return list()

