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
        output = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
        return output

    # also element-wise indepenent
    def backward(self: Tanh,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        output = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
        return dLoss_dModule * (1 - output * output)

    def parameters(self: Tanh) -> List[Parameter]:
        return list()

