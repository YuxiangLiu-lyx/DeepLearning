# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class TimeDistributed(Module):
    def __init__(self: TimeDistributed,
                 module: Module) -> None:
        self.module = module

    def forward(self: TimeDistributed,
                X: np.ndarray) -> np.ndarray:
        batch_size, seq_size = X.shape[:2]
        X_reshaped = X.reshape(batch_size * seq_size, *X.shape[2:])
        output = self.module.forward(X_reshaped)
        return output.reshape(batch_size, seq_size, *output.shape[1:])

    def backward(self: TimeDistributed,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        batch_size, seq_size = X.shape[:2]
        X_reshaped = X.reshape(batch_size * seq_size, *X.shape[2:])
        dLoss_reshaped = dLoss_dModule.reshape(batch_size * seq_size, *dLoss_dModule.shape[2:])
        dLoss_dX = self.module.backward(X_reshaped, dLoss_reshaped)
        return dLoss_dX.reshape(batch_size, seq_size, *dLoss_dX.shape[1:])

    def parameters(self: TimeDistributed) -> List[Parameter]:
        return self.module.parameters()
