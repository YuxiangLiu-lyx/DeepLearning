# SYSTEM IMPORTS
from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class Conv2d(Module):
    def __init__(self: Conv2d,
                 num_kernels: int,
                 num_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 padding: str = 'valid') -> None:
        self.num_kernels = num_kernels
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if isinstance(kernel_size, tuple):
            self.kernel_height = kernel_size[0]
            self.kernel_width = kernel_size[1]
        else:
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        fan_in = self.kernel_height * self.kernel_width * num_channels
        std = np.sqrt(2.0 / fan_in)
        self.W = Parameter(np.random.randn(self.kernel_height, self.kernel_width, num_channels, num_kernels) * std)
        self.b = Parameter(np.zeros(num_kernels))
        self.pad_amounts = self.get_pad_amounts()
   
   
    def get_pad_amounts(self: Conv2d) -> Tuple[int, int]:
        if self.padding == 'valid':
            return (0, 0)
        elif self.padding == 'same':
            pad_h = self.kernel_height // 2
            pad_w = self.kernel_width // 2
            return (pad_h, pad_w)
        else:
            raise ValueError(f"Invalid padding: {self.padding}")

   
    def pad_imgs(self: Conv2d, X: np.ndarray) -> np.ndarray:
        pad_h, pad_w = self.pad_amounts
        if pad_h == 0 and pad_w == 0:
            return X
        return np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0.0)

    
    def get_out_shape(self: Conv2d, X: np.ndarray) -> Tuple[int, int, int, int]:
        if isinstance(X, tuple):
            batch_size, height, width, channels = X
        else:
            batch_size, height, width, channels = X.shape
        pad_h, pad_w = self.pad_amounts
        out_height = (height - self.kernel_height + 2 * pad_h) // self.stride + 1
        out_width = (width - self.kernel_width + 2 * pad_w) // self.stride + 1
        return (batch_size, out_height, out_width, self.num_kernels)

    
    def forward(self: Conv2d, X: np.ndarray) -> np.ndarray:
        X_padded = self.pad_imgs(X)
        out_shape = self.get_out_shape(X)
        output = np.zeros(out_shape)
        batch_size, out_height, out_width, _ = out_shape
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride
                h_end = h_start + self.kernel_height
                w_start = w * self.stride
                w_end = w_start + self.kernel_width
                input_slice = X_padded[:, h_start:h_end, w_start:w_end, :]
                for k in range(self.num_kernels):
                    conv_sum = np.sum(input_slice * self.W.val[:, :, :, k], axis=(1, 2, 3))
                    output[:, h, w, k] = conv_sum + self.b.val[k]
        return output


    def backward(self: Conv2d, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        X_padded = self.pad_imgs(X)
        dLoss_dX = np.zeros_like(X_padded)
        dLoss_dW = np.zeros_like(self.W.val)
        dLoss_db = np.zeros_like(self.b.val)
        batch_size, out_height, out_width, _ = dLoss_dModule.shape
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride
                h_end = h_start + self.kernel_height
                w_start = w * self.stride
                w_end = w_start + self.kernel_width
                input_slice = X_padded[:, h_start:h_end, w_start:w_end, :]
                grad_out = dLoss_dModule[:, h, w, :]
                for k in range(self.num_kernels):
                    grad_k = grad_out[:, k]
                    grad_k_reshaped = grad_k.reshape(batch_size, 1, 1, 1)
                    dLoss_dX[:, h_start:h_end, w_start:w_end, :] += grad_k_reshaped * self.W.val[:, :, :, k]
                    dLoss_dW[:, :, :, k] += np.sum(grad_k_reshaped * input_slice, axis=0)
                    dLoss_db[k] += np.sum(grad_k)
        if not self.W.frozen:
            self.W.grad = dLoss_dW
        if not self.b.frozen:
            self.b.grad = dLoss_db
        pad_h, pad_w = self.pad_amounts
        if pad_h > 0 or pad_w > 0:
            h_start = pad_h if pad_h > 0 else 0
            h_end = -pad_h if pad_h > 0 else dLoss_dX.shape[1]
            w_start = pad_w if pad_w > 0 else 0
            w_end = -pad_w if pad_w > 0 else dLoss_dX.shape[2]
            dLoss_dX = dLoss_dX[:, h_start:h_end, w_start:w_end, :]
        return dLoss_dX


    def parameters(self: Conv2d) -> List[Parameter]:
        return [self.W, self.b]

