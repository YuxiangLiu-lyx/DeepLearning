from __future__ import annotations
from typing import List
from enum import Enum
import numpy as np

from ..module import Module
from ..param import Parameter


class SkipConnectionType(Enum):
    ADD = "add"
    CONCATENATE = "concatenate"


class SkipConnection(Module):
    def __init__(self: SkipConnection,
                 module: Module,
                 mode: SkipConnectionType = SkipConnectionType.ADD) -> None:
        self.module = module
        self.mode = mode

    def forward(self: SkipConnection,
                X: np.ndarray) -> np.ndarray:
        module_output = self.module.forward(X)
        
        if self.mode == SkipConnectionType.ADD:
            return X + module_output
        elif self.mode == SkipConnectionType.CONCATENATE:
            return np.concatenate([module_output, X], axis=-1)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def backward(self: SkipConnection,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        if self.mode == SkipConnectionType.ADD:
            dLoss_dModuleOutput = dLoss_dModule
            dLoss_dX_skip = dLoss_dModule
            dLoss_dX_module = self.module.backward(X, dLoss_dModuleOutput)
            return dLoss_dX_skip + dLoss_dX_module
        elif self.mode == SkipConnectionType.CONCATENATE:
            module_output = self.module.forward(X)
            output_dim = module_output.shape[-1]
            dLoss_dModuleOutput = dLoss_dModule[..., :output_dim]
            dLoss_dX_skip = dLoss_dModule[..., output_dim:]
            dLoss_dX_module = self.module.backward(X, dLoss_dModuleOutput)
            return dLoss_dX_skip + dLoss_dX_module
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def parameters(self: SkipConnection) -> List[Parameter]:
        return self.module.parameters()

