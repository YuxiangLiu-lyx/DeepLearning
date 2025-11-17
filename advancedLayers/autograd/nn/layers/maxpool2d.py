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
        self.pool_size: Tuple[int, int] = pool_size
        self.stride = stride

    def forward(self: MaxPool2d,
                X: np.ndarray) -> np.ndarray:
        batch_dim, h, w, num_channels = X.shape
        pool_h, pool_w = self.pool_size

        out_h: int = 1 + (h - pool_h) // self.stride
        out_w: int = 1 + (w - pool_w) // self.stride

        Y_hat: np.ndarray = np.zeros((batch_dim, out_h, out_w, num_channels))
        for i in range(out_h):
            for j in range(out_w):
                h_start_idx: int = i * self.stride
                h_end_idx: int = h_start_idx + pool_h

                w_start_idx: int = j * self.stride
                w_end_idx: int = w_start_idx + pool_w

                Y_hat[:, i, j, :] = np.max(X[:,h_start_idx:h_end_idx, w_start_idx:w_end_idx,:], axis=(1,2))
        return Y_hat


    # because we have learnable parameters here,
    # we need to do 3 things:
    #   1) compute dLoss_dW
    #   2) compute dLoss_db
    #   3) compute (and return) dLoss_dX
    def backward(self: MaxPool2d,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        dL_dX: np.ndarray = np.zeros_like(X)

        num_batches, out_h, out_w, num_channels = dLoss_dModule.shape
        pool_h, pool_w = self.pool_size

        for i in range(out_h):
            for j in range(out_w):
                h_start_idx: int = i * self.stride
                h_end_idx: int = h_start_idx + pool_h

                w_start_idx: int = j * self.stride
                w_end_idx: int = w_start_idx + pool_w

                # get the patch we used to compute this output cell
                X_patch: np.ndarray = X[:,h_start_idx:h_end_idx, w_start_idx:w_end_idx,:]

                # a binary mask which will index out which examples contributed to the max
                # value recorded in the output
                mask = np.zeros_like(X_patch)
                _, h, w, _ = X_patch.shape

                X_patch = X_patch.reshape(num_batches, h*w, num_channels)
                batch_idx, channel_idx = np.indices([num_batches, num_channels])

                # print("\t", i, j, f"{h_start_idx}->{h_end_idx}", f"{w_start_idx}->{w_end_idx}", X_patch)

                mask.reshape(X_patch.shape)[batch_idx, np.argmax(X_patch, axis=1), channel_idx] = 1

                # print(dL_dX[:, height_start_idx:height_end_idx,
                #             width_start_idx:width_end_idx, :].shape,
                #       mask.shape,
                #       dLoss_dModule[:,i:i+1, j:j+1, :].shape)
                # use the mask to distribute dLoss_dModule to the corresponding values of X
                dL_dX[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx,:] += dLoss_dModule[:,i:i+1, j:j+1, :] * mask

        return dL_dX

    def parameters(self: MaxPool2d) -> List[Parameter]:
        return list()
