# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from ..lf import LossFunction


# TYPES DECLARED IN THIS MODULE


class BinaryCrossEntropy(LossFunction):

    # TODO: compute the expected binary cross entropy loss!
    def forward(self: BinaryCrossEntropy,
                Y_hat: np.ndarray,
                Y_gt: np.ndarray) -> float:
        assert(Y_hat.shape == Y_gt.shape)
        return (np.sum(-np.log(Y_hat[Y_gt!=0])) +\
               np.sum(-np.log(1-Y_hat[Y_gt==0])))/(Y_hat.shape[0])

    # TODO: take the derivative of binary cross entropy with respect to Y_hat
    # you will then need to compute this derivative and pass return it.
    def backward(self: BinaryCrossEntropy,
                 Y_hat: np.ndarray,
                 Y_gt: np.ndarray) -> np.ndarray:
        assert(Y_hat.shape == Y_gt.shape)

        dL_dYhat = np.zeros_like(Y_hat)
        dL_dYhat[Y_gt!=0] += -1.0/(Y_hat[Y_gt!=0])
        dL_dYhat[Y_gt==0] += 1.0/(1-Y_hat[Y_gt==0])

        dL_dYhat /= Y_hat.shape[0]
        return dL_dYhat

