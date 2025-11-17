# SYSTEM IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from typing import List

# PYTHON PROJECT IMPORTS
from nn.models.sequential import Sequential
from nn.layers.dense import Dense
from nn.layers.tanh import Tanh
from nn.layers.sigmoid import Sigmoid
from nn.layers.relu import ReLU
from nn.optimizers.sgd import SGDOptimizer
from nn.losses.mse import MeanSquaredError as MSE
from nn.param import Parameter
from nn.module import Module
from nn.lf import LossFunction

def grad_check(X: np.ndarray, Y_gt: np.ndarray, m: Module, loss_func: LossFunction,
               epsilon: float = 1e-4, delta: float = 1e-6) -> None:
    print("--- Running Gradient Check ---")
    
    params: List[Parameter] = m.parameters()
    analytical_grads: List[np.ndarray] = [p.grad.copy() for p in params]
    numerical_grads: List[np.ndarray] = [np.zeros_like(p.val) for p in params]
    
    for p, num_grad in zip(params, numerical_grads):
        for index, _ in np.ndenumerate(p.val):
            original_val = p.val[index]
            p.val[index] = original_val + epsilon
            loss_plus = loss_func.forward(m.forward(X), Y_gt)
            p.val[index] = original_val - epsilon
            loss_minus = loss_func.forward(m.forward(X), Y_gt)
            num_grad[index] = (loss_plus - loss_minus) / (2 * epsilon)
            p.val[index] = original_val

    ratios = np.array([np.linalg.norm(ag - ng) / (np.linalg.norm(ag + ng) + epsilon)
                       for ag, ng in zip(analytical_grads, numerical_grads)], dtype=float)

    if np.any(ratios > delta):
        raise RuntimeError(f"ERROR: Gradient check failed. Delta: [{delta}], Ratios: {ratios}")
    else:
        print(f"Delta: [{delta}], Ratios: {ratios}")
        print(f"--- Gradient Check Passed! (Max ratio: {np.max(ratios):.2e}) ---")


if __name__ == "__main__":
    in_dim: int = 100
    out_dim: int = 3
    lr: float = 0.01
    max_epochs: int = 1000

    X, Y_gt_unscaled = make_regression(n_samples=500,
                                       n_features=in_dim,
                                       n_targets=out_dim,
                                       noise=0.1,
                                       random_state=42)
    scaler = MinMaxScaler()
    Y_gt = scaler.fit_transform(Y_gt_unscaled)

    m: Sequential = Sequential()
    m.add(Dense(in_dim, 20))
    m.add(Tanh())
    m.add(Dense(20, out_dim))
    m.add(ReLU())
    m.add(Sigmoid())

    optim: SGDOptimizer = SGDOptimizer(m.parameters(), lr)
    loss_func: MSE = MSE()

    Y_hat_check = m.forward(X)
    initial_grad = loss_func.backward(Y_hat_check, Y_gt)
    optim.reset()
    m.backward(X, initial_grad)
    grad_check(X, Y_gt, m, loss_func)

    losses = []
    print("\n--- Starting Training ---")
    for i in range(max_epochs):
        optim.reset()
        Y_hat = m.forward(X)
        loss = loss_func.forward(Y_hat, Y_gt)
        losses.append(loss)
        grad_from_loss = loss_func.backward(Y_hat, Y_gt)
        m.backward(X, grad_from_loss)
        optim.step()
        
        if i % 100 == 0:
            print(f"Epoch {i:4d}/{max_epochs}, Loss: {loss:.6f}")

    print("--- Training Finished ---")

    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.show()
