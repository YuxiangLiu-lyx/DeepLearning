# SYSTEM IMPORTS
from __future__ import annotations
from typing import List, Tuple
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class MaxPool2d(Module):
    def __init__(self: MaxPool2d,
                 pool_size: Tuple[int, int],
                 stride: int = 2) -> None:
        self.pool_size = pool_size
        self.stride = stride
        self.pool_h = pool_size[0]
        self.pool_w = pool_size[1]

    def forward(self: MaxPool2d,
                X: np.ndarray) -> np.ndarray:
        num_examples, h, w, num_channels = X.shape
        out_h = 1 + (h - self.pool_h) // self.stride
        out_w = 1 + (w - self.pool_w) // self.stride
        output = np.zeros((num_examples, out_h, out_w, num_channels))
        for out_h_idx in range(out_h):
            for out_w_idx in range(out_w):
                h_start_idx = out_h_idx * self.stride
                h_end_idx = h_start_idx + self.pool_h
                w_start_idx = out_w_idx * self.stride
                w_end_idx = w_start_idx + self.pool_w
                X_pool_all_channels = X[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :]
                output[:, out_h_idx, out_w_idx, :] = np.max(X_pool_all_channels, axis=(1, 2))
        
        return output

    def backward(self: MaxPool2d,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        out_h, out_w = dLoss_dModule.shape[1], dLoss_dModule.shape[2]
        dLoss_dX = np.zeros_like(X)
        for out_h_idx in range(out_h):
            for out_w_idx in range(out_w):
                h_start_idx = out_h_idx * self.stride
                h_end_idx = h_start_idx + self.pool_h
                w_start_idx = out_w_idx * self.stride
                w_end_idx = w_start_idx + self.pool_w
                X_pool = X[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :]
                max_vals = np.max(X_pool, axis=(1, 2), keepdims=True)
                pool_mask = (X_pool == max_vals).astype(float)
                dLoss_dX[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :] += \
                    pool_mask * dLoss_dModule[:, out_h_idx:out_h_idx+1, out_w_idx:out_w_idx+1, :]
        
        return dLoss_dX

    def parameters(self: MaxPool2d) -> List[Parameter]:
        return []

