# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from ..lf import LossFunction


# TYPES DECLARED IN THIS MODULE


class BinaryCrossEntropy(LossFunction):

    def forward(self: BinaryCrossEntropy,
                Y_hat: np.ndarray,
                Y_gt: np.ndarray) -> float:
        assert(Y_hat.shape == Y_gt.shape)
        n_samples = Y_hat.shape[0]
        loss = -np.sum(Y_gt * np.log(Y_hat) + (1 - Y_gt) * np.log(1 - Y_hat)) / n_samples
        return loss

    def backward(self: BinaryCrossEntropy,
                 Y_hat: np.ndarray,
                 Y_gt: np.ndarray) -> np.ndarray:
        assert(Y_hat.shape == Y_gt.shape)
        n_samples = Y_hat.shape[0]
        gradient = (Y_hat - Y_gt) / (Y_hat * (1. - Y_hat) * n_samples)
        return gradient