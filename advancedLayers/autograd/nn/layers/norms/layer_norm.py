from __future__ import annotations
from typing import List
import numpy as np

from ...module import Module
from ...param import Parameter


class LayerNorm(Module):
    def __init__(self: LayerNorm,
                 gamma: float = 1.0,
                 beta: float = 0.0,
                 epsilon: float = 1e-5) -> None:
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon

    def forward(self: LayerNorm,
                X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        X_normalized = (X - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * X_normalized + self.beta

    def jacobian_single_example(self: LayerNorm,
                                x: np.ndarray,
                                y_hat: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        mean = np.mean(x)
        var = np.var(x)
        std = np.sqrt(var + self.epsilon)
        
        x_norm = (x - mean) / std
        
        jacobian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    d_norm = 1.0 / std
                else:
                    d_norm = 0.0
                d_mean = -1.0 / (n * std)
                d_var = -x_norm[i] / (2 * n * std)
                
                d_total = d_norm + d_mean + d_var * 2 * (x[j] - mean)
                jacobian[i, j] = self.gamma * d_total
        
        return jacobian

    def backward(self: LayerNorm,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        Y_hat = self.forward(X)
        
        dYhat_dX = np.zeros(Y_hat.shape + (Y_hat.shape[-1],))
        for example_idx, (x, y_hat) in enumerate(zip(X, Y_hat)):
            dYhat_dX[example_idx] = self.jacobian_single_example(x, y_hat)
        
        return np.einsum("...jk, ...kl", dYhat_dX,
            np.expand_dims(dLoss_dModule, axis=-1)).reshape(X.shape)

    def parameters(self: LayerNorm) -> List[Parameter]:
        return list()

