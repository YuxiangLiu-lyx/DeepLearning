from __future__ import annotations
from ..optimizer import Optimizer
from ..param import Parameter

class SGDOptimizer(Optimizer):
    def step_parameter(self: SGDOptimizer, P: Parameter) -> None:
            P.val = P.val - self.lr * P.grad
