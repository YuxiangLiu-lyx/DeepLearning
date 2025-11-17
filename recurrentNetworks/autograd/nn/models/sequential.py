# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class Sequential(Module):
    def __init__(self: Sequential,
                 layers: List[Module] = None) -> None:
        self.layers: List[Module] = layers
        if layers is None:
            self.layers = list()

    def add(self: Sequential,
            m: Module) -> Sequential:
        self.layers.append(m)
        return self

    def parameters(self: Sequential) -> List[Parameter]:
        params: List[Parameters] = list()
        for m in self.layers:
            params.extend(m.parameters())
        return params

    def forward(self: Sequential,
                X: np.ndarray) -> np.ndarray:
        for m in self.layers:
            X = m.forward(X)
        return X

    def backward(self: Sequential,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        Xs: List[np.ndarray] = [X]
        for m in self.layers:
            X = m.forward(X)
            Xs.append(X)

        for i,m in enumerate(self.layers[::-1]):
            dLoss_dModule = m.backward(Xs[-i-2], dLoss_dModule)
        return dLoss_dModule

