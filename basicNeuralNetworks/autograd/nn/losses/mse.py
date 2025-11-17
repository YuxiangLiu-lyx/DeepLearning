# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from ..lf import LossFunction


# TYPES DECLARED IN THIS MODULE


class MeanSquaredError(LossFunction):

    def forward(self: MeanSquaredError,
                Y_hat: np.ndarray,
                Y_gt: np.ndarray) -> float:
        assert(Y_hat.shape == Y_gt.shape)
        return np.sum((Y_hat - Y_gt)**2) / (2*Y_hat.shape[0])

    def backward(self: MeanSquaredError,
                 Y_hat: np.ndarray,
                 Y_gt: np.ndarray) -> np.ndarray:
        assert(Y_hat.shape == Y_gt.shape)
        return (Y_hat - Y_gt) / Y_hat.shape[0]

