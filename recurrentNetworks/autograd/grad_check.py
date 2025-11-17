# SYSTEM IMPORTS
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from nn.module import Module
from nn.param import Parameter
from nn.lf import LossFunction


def grad_check(X: np.ndarray,
               Y_gt: np.ndarray,
               m: Module,
               ef: LossFunction,
               epsilon: float = 1e-4,
               delta: float = 1e-6) -> None:
    params: List[Parameter] = [P for P in m.parameters() if not P.frozen]
    num_grads: List[np.ndarray] = [np.zeros_like(P.val) for P in params]
    sym_grads: List[np.ndarray] = [P.grad for P in params]

    for P, N in zip(params, num_grads):
        for index, v in np.ndenumerate(P.val):
            P.val[index] += epsilon
            N[index] += ef.forward(m.forward(X), Y_gt)
            P.val[index] -= 2*epsilon
            N[index] -= ef.forward(m.forward(X), Y_gt)
            P.val[index] = v
            N[index] /= (2*epsilon)

    ratios: np.ndarray = np.array([np.linalg.norm(SG-NG)/
                                   np.linalg.norm(SG+NG)
                                   for SG, NG in zip(sym_grads, num_grads)], dtype=float)
    if np.sum(ratios > delta) > 0:
        raise RuntimeError("ERROR: failed grad check. delta: [%s], ratios: %s"
            % (delta, ratios))

