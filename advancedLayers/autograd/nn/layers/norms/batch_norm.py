from __future__ import annotations
from typing import List
import numpy as np

from ...module import Module
from ...param import Parameter


class BatchNorm(Module):
    def __init__(self: BatchNorm,
                 gamma: float = 1.0,
                 beta: float = 0.0,
                 epsilon: float = 1e-5) -> None:
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon

    def forward(self: BatchNorm,
                X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=0, keepdims=True)
        var = np.var(X, axis=0, keepdims=True)
        X_normalized = (X - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * X_normalized + self.beta

    def jacobian_single_feature(self: BatchNorm,
                                 x_feature: np.ndarray,
                                 y_hat_feature: np.ndarray) -> np.ndarray:
        n = x_feature.shape[0]
        mean = np.mean(x_feature)
        var = np.var(x_feature)
        std = np.sqrt(var + self.epsilon)
        
        x_norm = (x_feature - mean) / std
        
        jacobian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    d_norm = 1.0 / std
                else:
                    d_norm = 0.0
                d_mean = -1.0 / (n * std)
                d_var = -x_norm[i] / (2 * n * std)
                
                d_total = d_norm + d_mean + d_var * 2 * (x_feature[j] - mean)
                jacobian[i, j] = self.gamma * d_total
        
        return jacobian

    def backward(self: BatchNorm,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        Y_hat = self.forward(X)
        
        dLoss_dX = np.zeros_like(X)
        
        num_features = X.shape[-1]
        for feature_idx in range(num_features):
            x_feature = X[:, feature_idx]
            y_hat_feature = Y_hat[:, feature_idx]
            dLoss_dModule_feature = dLoss_dModule[:, feature_idx]
            
            jacobian = self.jacobian_single_feature(x_feature, y_hat_feature)
            
            dLoss_dX[:, feature_idx] = np.dot(jacobian.T, dLoss_dModule_feature)
        
        return dLoss_dX

    def parameters(self: BatchNorm) -> List[Parameter]:
        return list()

