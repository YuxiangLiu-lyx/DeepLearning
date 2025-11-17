# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class ReLU(Module):

    # the formula of sigmoid is 1/(1 + e^-x)
    def forward(self: ReLUType,
                X: np.ndarray) -> np.ndarray:
        return np.clip(X, 0, None)


    # SINCE sigmoid is element-wise INDEPENDENT, this makes derivatives nice.
    # lets say that X is a vector (x_1, x_2, ..., x_n)
    # so sigmoid(x) is also a vector (y_1, y_2, ..., y_n)

    # if I take the dsigmoid(x)/dx then im really computing a jacobian matrix:
    #               y_1     y_2     ...     y_n
    #      x_1  dy_1/dx_1 dy_2/dx_2       dy_n/dx_n
    #      x_2
    #      ...
    #      x_n

    # BECAUSE of element-wise independence, this jacobian is diagonal

    # derivative of sigmoid is sigmoid(x)(1-sigmoid(x))
    def backward(self: ReLUType,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        return (X > 0) * dLoss_dModule

    def parameters(self: ReLUType) -> List[Parameter]:
        return list()

