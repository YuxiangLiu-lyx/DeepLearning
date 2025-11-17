# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class Flatten(Module):
    def __init__(self: Flatten) -> None:
        ...

    def forward(self: Flatten,
                X: np.ndarray) -> np.ndarray:
        return X.ravel().reshape(X.shape[0], -1)

    # because we have learnable parameters here,
    # we need to do 3 things:
    #   1) compute dLoss_dW
    #   2) compute dLoss_db
    #   3) compute (and return) dLoss_dX
    def backward(self: Flatten,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        return dLoss_dModule.reshape(X.shape)

    def parameters(self: Flatten) -> List[Parameter]:
        return list()

