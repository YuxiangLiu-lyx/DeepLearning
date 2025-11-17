# SYSTEM IMPORTS
from __future__ import annotations
from abc import abstractmethod, ABC
import numpy as np


# PYTHON PROJECT IMPORTS



# TYPES DECLARED IN THIS MODEUL


class LossFunction(ABC):

    @abstractmethod
    def forward(self: LossFunction,
                Y_hat: np.ndarray,
                Y_gt: np.ndarray) -> float:
        ...

    @abstractmethod
    def backward(self: LossFunction,
                 Y_hat: np.ndarray,
                 Y_gt: np.ndarray) -> np.ndarray:
        ...

