# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class Sigmoid(Module):

    def forward(self: Sigmoid,
                X: np.ndarray) -> np.ndarray:
        output = 1. / (1. + np.exp(-X))
        return output

    # also element-wise indepenent
    def backward(self: Sigmoid,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        output = 1. / (1. + np.exp(-X))
        gradient = output * (1. - output)
        return dLoss_dModule * gradient

    def parameters(self: Sigmoid) -> List[Parameter]:
        return list()

