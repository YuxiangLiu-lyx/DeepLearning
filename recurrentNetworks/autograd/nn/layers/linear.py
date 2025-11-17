# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class Linear(Module):

    def forward(self: Linear,
                X: np.ndarray) -> np.ndarray:
        return X

    # also element-wise indepenent
    def backward(self: Linear,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        return dLoss_dModule

    def parameters(self: Linear) -> List[Parameter]:
        return list()

