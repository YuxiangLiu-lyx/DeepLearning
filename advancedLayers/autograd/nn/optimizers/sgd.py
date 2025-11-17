# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from ..optimizer import Optimizer
from ..param import Parameter


# TYPES DECLARD IN THIS MODULE


class SGDOptimizer(Optimizer):

    def step_parameter(self: SGDOptimizer,
                       P: Parameter) -> None:
        P.step(self.lr * P.grad)

