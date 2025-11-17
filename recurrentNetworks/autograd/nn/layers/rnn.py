from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np


from ..module import Module
from ..param import Parameter
from .cells.rnn_cell import RNNCell


class RNN(Module):
    def __init__(self: RNN,
                 cell: RNNCell,
                 return_sequences: bool = False,
                 return_states: bool = False,
                 backprop_through_time_limit: int = None) -> None:
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_states = return_states
        self.backprop_through_time_limit = backprop_through_time_limit if backprop_through_time_limit is not None else np.inf

    def init_states(self: RNN,
                    batch_size: int) -> np.ndarray:
        return self.cell.init_states(batch_size)

    def forward(self: RNN,
                X: np.ndarray,
                states_init: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        batch_size, seq_size = X.shape[:2]
        if states_init is None:
            states_init = self.init_states(batch_size)
        
        states = [states_init]
        predictions = []
        
        for t in range(seq_size):
            H_t, A_t = self.cell.forward(states[-1], X[:, t])
            states.append(H_t)
            predictions.append(A_t)
        
        if self.return_sequences:
            if self.return_states:
                states_arr = np.array(states).transpose(1, 0, 2)
                pred_arr = np.array(predictions).transpose(1, 0, 2)
                return states_arr, pred_arr
            else:
                return np.array(predictions).transpose(1, 0, 2)
        else:
            if self.return_states:
                return states[-1], predictions[-1]
            else:
                return predictions[-1]

    def backward(self: RNN,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray,
                 states_init: np.ndarray = None) -> np.ndarray:
        batch_size, seq_size = X.shape[:2]
        if states_init is None:
            states_init = self.init_states(batch_size)
        
        states = [states_init] 
        for t in range(seq_size):
            H_t, A_t = self.cell.forward(states[-1], X[:, t])
            states.append(H_t)
        
        dLoss_dX = np.zeros_like(X)
        dLoss_dStates = np.zeros_like(states[-1])
        
        if self.return_sequences:
            for t in range(seq_size - 1, -1, -1):
                dLoss_dModule_t = dLoss_dModule[:, t] 
                
                dLoss_dH_t_minus_1, dLoss_dX_t = self.cell.backward(states[t], X[:, t], dLoss_dModule_t, dLoss_dStates)
                
                dLoss_dX[:, t] = dLoss_dX_t
                dLoss_dStates = dLoss_dH_t_minus_1
                
                if t <= seq_size - 1 - self.backprop_through_time_limit:
                    break
        else:
            dLoss_dModule_t = dLoss_dModule
            
            for t in range(seq_size - 1, -1, -1): 
                
                dLoss_dH_t_minus_1, dLoss_dX_t = self.cell.backward(states[t], X[:, t], dLoss_dModule_t, dLoss_dStates)
                
                dLoss_dX[:, t] = dLoss_dX_t
                dLoss_dStates = dLoss_dH_t_minus_1
                dLoss_dModule_t = None 
                
                if t <= seq_size - 1 - self.backprop_through_time_limit:
                    break
        
        return dLoss_dX

    def parameters(self: RNN) -> List[Parameter]:
        return self.cell.parameters()