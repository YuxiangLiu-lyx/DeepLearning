from __future__ import annotations
from ..module import Module
from ..param import Parameter
from typing import List
import numpy as np

class Sequential(Module):
    def __init__(self, layers: List = None) -> None:
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.inputs = []

    def add(self, layer : Module) -> None:
        self.layers.append(layer)

    def forward(self: Module,
                X: np.ndarray) -> np.ndarray:
        self.inputs = []
        output = X
        for layer in self.layers:
            self.inputs.append(output)
            output = layer.forward(output)
        return output

    def backward(self: Module,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        grad = dLoss_dModule
        for layer, layer_input in zip(reversed(self.layers), reversed(self.inputs)):
            grad = layer.backward(layer_input, grad)
        return grad

    def parameters(self: Module) -> List[Parameter]:
        all_params = []
        for layer in self.layers:
            if len(layer.parameters()) > 0:
                all_params.extend(layer.parameters())
        return all_params
            