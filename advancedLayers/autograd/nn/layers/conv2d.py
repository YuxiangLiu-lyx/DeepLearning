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
                 padding: str = "valid") -> None:
        self.num_kernels: int = int(num_kernels)
        self.num_channels: int = int(num_channels)

        # self.kernel size will be [height, width]
        # however if the input is a scalar we assume height = width
        self.kernel_size: Tuple[int, int] = None
        if not isinstance(kernel_size, tuple):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            assert(len(kernel_size) == 2)
            self.kernel_size = tuple(kernel_size)

        self.stride: int = int(stride)
        self.padding: str = padding

        kernel_shapes: Tuple[int, int, int] = self.kernel_size + (self.num_channels,)
        self.W: Parameter = Parameter(np.random.randn(*kernel_shapes, num_kernels))
        self.b: Parameter = Parameter(np.random.randn(self.num_kernels))

        self.pad_amounts: Tuple[int, int] = self.get_pad_amounts()


    def get_pad_amounts(self: Conv2d) -> Tuple[int, int]:
        padding_amount: Tuple[int, int] = None
        if isinstance(self.padding, str) and self.padding.lower() == "same":
            # gonna add kernel height and width amount (half to each side)
            padding_amount = ((self.W.val.shape[0])//2, (self.W.val.shape[1])//2) # TODO: put back the -1 before //2
        elif isinstance(self.padding, str) and self.padding.lower() == "valid":
            # no padding
            padding_amount = (0,0)
        else:
            raise ValueError(f"[ERROR] Conv2d.get_pad_amounts: padding must be one of {'valid', 'same'}, not {self.padding}")

        return padding_amount


    def pad_imgs(self: Conv2d,
                 X: np.ndarray) -> np.ndarray:
        # computes the new dimensions of the padded examples,
        # applies the padding (i.e. adds zeros around the examples)
        # and returns the padded examples
        # padding_amount: Tuple[int, int] = self.compute_pad_buffer()

        # expect examples to have shape [batch_dim, height, width, num_channels]
        # thankfully numpy has a handy function for padding which we can use
        return np.pad(X,
                      [(0, 0),                                      # pad amounts for axis 0
                       (self.pad_amounts[0], self.pad_amounts[0]),  # pad amounts for axis 1
                       (self.pad_amounts[1], self.pad_amounts[1]),  # pad amounts for axis 2
                       (0, 0)],                                     # pad amounts for axis 3
                      mode="constant",
                      constant_values=0.0)

    def get_out_shape(self: Conv2d,
                      X_shape: np.ndarray) -> Tuple[int, int, int, int]:
        num_examples, h, w, num_channels = X_shape
        kernel_h, kernel_w, kernel_channels, num_filters = self.W.val.shape

        """
        out_shape: Tuple[int, int, int, int] = None
        if isinstance(self.padding, str) and self.padding.lower() == "same":
            out_shape = (num_examples, h, w, num_filters,)
        elif isinstance(self.padding, str) and self.padding.lower() == "valid":
            out_shape = (num_examples, (h - kernel_h + 2 * self.pad_amounts[0]) // self.stride + 1,
                                       (w - kernel_w + 2 * self.pad_amounts[1]) // self.stride + 1,
                         num_filters,)
        else:
            raise ValueError(f"[ERROR] Conv2d.get_pad_amounts: padding must be one of {'valid', 'same'}, not {self.padding}")
        """

        out_shape = (num_examples, (h - kernel_h + 2 * self.pad_amounts[0]) // self.stride + 1,
                                   (w - kernel_w + 2 * self.pad_amounts[1]) // self.stride + 1,
                     num_filters,)
        return out_shape


    def forward(self: Conv2d,
                X: np.ndarray) -> np.ndarray:
        batch_dim, h, w, num_channels = X.shape
        kernel_h, kernel_w, kernel_channels, num_filters = self.W.val.shape

        # pad the input volume
        X_padded: np.ndarray = self.pad_imgs(X)

        # get the output shape and pre-allocated the volume
        out_shape: Tuple[int, int, int, int] = self.get_out_shape(X.shape)
        _, out_h, out_w, _ = out_shape
        Y_hat: np.ndarray = np.zeros(out_shape, dtype=float)

        for i in range(out_h):
            for j in range(out_w):
                h_start_idx: int = i*self.stride
                h_end_idx: int = h_start_idx + kernel_h

                w_start_idx: int = j*self.stride
                w_end_idx: int = w_start_idx + kernel_w

                # print("\t", h_start_idx + "->" + h_end_idx, w_start_idx + "->" + w_end_idx)

                Y_hat[:, i, j, :] = (X_padded[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :, np.newaxis] *
                    self.W.val[np.newaxis, :, :, :]).sum(axis=(1,2,3))

        return Y_hat + self.b.val.reshape(1,1,1,-1)

    def backward(self: Conv2d,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        batch_size, h, w, num_channels = X.shape
        kernel_h, kernel_w, kernel_channels, num_filters = self.W.val.shape

        pad_h, pad_w = self.pad_amounts # self.compute_pad_buffer()
        _, out_h, out_w, _ = dLoss_dModule.shape

        X_padded: np.ndarray = self.pad_imgs(X)
        dL_dX: np.ndarray = np.zeros_like(X_padded, dtype=float)

        """"""
        self.b.grad += np.sum(dLoss_dModule, axis=(0,1,2))
        for i in range(out_h):
            for j in range(out_w):
                h_start_idx: int = i*self.stride
                h_end_idx: int = h_start_idx + kernel_h

                w_start_idx: int = j*self.stride
                w_end_idx: int = w_start_idx + kernel_w

                dL_dX[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :] += \
                    (self.W.val[np.newaxis, :, :, :, :] * dLoss_dModule[:, i:i+1, j:j+1, np.newaxis, :]).sum(axis=4)
                self.W.grad += (
                    X_padded[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :, np.newaxis] *
                    dLoss_dModule[:, i:i+1, j:j+1, np.newaxis, :]
                ).sum(axis=0)
        """"""
        """
        # for each output pixel
        for i in range(out_h):
            for j in range(out_w):

                # compute the location of the patch that producted the output pixel
                h_start_idx: int = i*self.stride
                h_end_idx: int = h_start_idx + k_height

                w_start_idx: int = j*self.stride
                w_end_idx: int = w_start_idx + kernel_w

                # for each kernel (filter), update the gradient of that kernel
                for l in range(self.num_kernels):
                    self.W.grad[:,:,:,l] += np.sum(X_padded[:,h_start_idx:h_end_idx, w_start_idx:w_end_idx, :] *
                                                   dLoss_dModule[:, i:i+1, j:j+1, l:l+1],
                                                   axis=0)

                    # for each example, update the gradient of the patch for that example
                    # according to a specific kernel
                    for example_idx in range(batch_size):
                        dL_dX[example, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :] += self.W.val[:, :, :, l] *\
                            dLoss_dModule[example_idx, i:i+1, j:j+1, l:l+1]
        self.b.grad += np.sum(dLoss_dModule, axis=(0,1,2))
        """

        # slice out derivatives for elements that aren't part of the padding
        return dL_dX[:, pad_h:pad_h + h, pad_w:pad_w + w, :]

    def parameters(self: Conv2d) -> List[Parameter]:
        return [self.W, self.b]

