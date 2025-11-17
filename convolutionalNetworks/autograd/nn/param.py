# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS


# TYPES DECLARED IN THIS MODULE


class Parameter(object):
    def __init__(self: Parameter,
                 V: np.ndarray,
                 frozen: bool = False) -> None:
        self.val: np.ndarray = V
        self.grad: np.ndarray = None
        self.frozen: bool = frozen

    def freeze(self: Parameter) -> None:
        self.frozen = True

    def thaw(self: Parameter) -> None:
        self.frozen = False

    def reset(self: Parameter) -> Parameter:
        self.grad = np.zeros_like(self.val)
        return self

    def step(self: Parameter,
             G: np.ndarray) -> Parameter:
        self.val -= G # np.clip(G, -100, 100)
        return self.reset()

