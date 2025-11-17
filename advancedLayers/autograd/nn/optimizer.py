# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
from abc import abstractmethod, ABC
import numpy as np


# PYTHON PROJECT IMPORTS
from .param import Parameter


# TYPES DECLARED IN THIS MODULE


class Optimizer(ABC):
    def __init__(self: Optimizer,
                 parameters: List[Parameter],
                 lr: float) -> None:
        self.parameters: List[Parameter] = parameters
        self.lr: float = lr

    def reset(self: Optimizer) -> None:
        for p in self.parameters:
            p.reset()

    def step(self: Optimizer) -> None:
        for P in self.parameters:
            if not P.frozen:
                self.step_parameter(P)

    @abstractmethod
    def step_parameter(self: Optimizer,
                       P: Parameter) -> None:
        ...

