# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from ..lf import LossFunction


# TYPES DECLARED IN THIS MODULE


class CategoricalCrossEntropy(LossFunction):
    delta = 1e-12
    def forward(self: CategoricalCrossEntropy,
                Y_hat: np.ndarray,
                Y_gt: np.ndarray) -> float:
        assert(Y_hat.shape == Y_gt.shape)
        return -np.sum(Y_gt * np.log(Y_hat + self.delta)) / Y_hat.shape[0]

    def backward(self: CategoricalCrossEntropy,
                 Y_hat: np.ndarray,
                 Y_gt: np.ndarray) -> np.ndarray:
        assert(Y_hat.shape == Y_gt.shape)
        return -Y_gt / (Y_hat + self.delta) / Y_hat.shape[0] 

