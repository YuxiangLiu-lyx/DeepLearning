from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np


from .rnn_cell import RNNCell
from ...module import Module
from ...param import Parameter
from ..dense import Dense
from ..tanh import Tanh
from ..sigmoid import Sigmoid


class VanillaRNNCell(RNNCell):
    def __init__(self: VanillaRNNCell,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 hidden_activation: Module = None,
                 output_activation: Module = None) -> None:
        super().__init__(in_dim, hidden_dim)
        self.out_dim = out_dim
        
        self.hidden_activation = Tanh() if hidden_activation is None else hidden_activation
        self.output_activation = Sigmoid() if output_activation is None else output_activation
        
        self.hidden_dense = Dense(in_dim + hidden_dim, hidden_dim)
        self.out_dense = Dense(hidden_dim, out_dim)

    def init_states(self: VanillaRNNCell,
                    batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.hidden_dim))

    def forward(self: VanillaRNNCell,
                X_t: np.ndarray,
                H_t_minus_1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        concat_input = np.concatenate([H_t_minus_1, X_t], axis=1)
        
        Z_t = self.hidden_dense.forward(concat_input)
        H_t = self.hidden_activation.forward(Z_t)
        R_t = self.out_dense.forward(H_t)
        A_t = self.output_activation.forward(R_t)
        
        return H_t, A_t

    def backward(self: VanillaRNNCell,
                 H_t_minus_1: np.ndarray,
                 X_t: np.ndarray,
                 dLoss_dModule_t: np.ndarray,
                 dLoss_dStates_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        concat_input = np.concatenate([X_t, H_t_minus_1], axis=1)
        
        Z_t = self.hidden_dense.forward(concat_input)
        H_t = self.hidden_activation.forward(Z_t)
        
        if self.hidden_dense.W.grad is None:
            self.hidden_dense.W.grad = np.zeros_like(self.hidden_dense.W.val)
            self.hidden_dense.b.grad = np.zeros_like(self.hidden_dense.b.val)
        if self.out_dense.W.grad is None:
            self.out_dense.W.grad = np.zeros_like(self.out_dense.W.val)
            self.out_dense.b.grad = np.zeros_like(self.out_dense.b.val)

        dLoss_dH_t = np.zeros_like(H_t)
        
        if dLoss_dModule_t is not None:
            R_t = self.out_dense.forward(H_t)
            dLoss_dR_t = self.output_activation.backward(R_t, dLoss_dModule_t)
            dLoss_dH_t += self.out_dense.backward(H_t, dLoss_dR_t)
            
        if dLoss_dStates_t is not None:
            dLoss_dH_t += dLoss_dStates_t
        
        dLoss_dZ_t = self.hidden_activation.backward(Z_t, dLoss_dH_t)
        dLoss_dConcat = self.hidden_dense.backward(concat_input, dLoss_dZ_t)
        
        dLoss_dX_t = dLoss_dConcat[:, :self.in_dim]
        dLoss_dH_t_minus_1 = dLoss_dConcat[:, self.in_dim:]
        
        return dLoss_dH_t_minus_1, dLoss_dX_t

    def parameters(self: VanillaRNNCell) -> List[Parameter]:
        params = self.hidden_dense.parameters()
        params += self.hidden_activation.parameters()
        params += self.out_dense.parameters()
        params += self.output_activation.parameters()
        return params