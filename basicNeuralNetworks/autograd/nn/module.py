# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
from abc import abstractmethod, ABC
import numpy as np


# PYTHON PROJECT IMPORTS
from .param import Parameter


# TYPES DECLARED IN THIS MODULE


class Module(ABC):

    @abstractmethod
    def forward(self: Module,
                X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self: Module,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def parameters(self: Module) -> List[Parameter]:
        ...

    def freeze(self: Module) -> None:
        for P in self.parameters():
            P.freeze()

    def thaw(self: Module) -> None:
        for P in self.parameters():
            P.thaw()

