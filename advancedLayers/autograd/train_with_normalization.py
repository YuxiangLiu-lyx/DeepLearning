from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from nn.module import Module
from nn.lf import LossFunction
from nn.layers.dense import Dense
from nn.layers.sigmoid import Sigmoid
from nn.layers.tanh import Tanh
from nn.layers.norms.batch_norm import BatchNorm
from nn.layers.norms.layer_norm import LayerNorm
from nn.models.sequential import Sequential
from nn.optimizers.sgd import SGDOptimizer as SGD
from nn.losses.mse import MeanSquaredError as MSE


from grad_check import grad_check


np.random.seed(12345)



lr: float = 1e-5
max_epochs: int = 1000

in_dim: int = 100
out_dim: int = 3
num_examples: int =  200

X: np.ndarray = np.random.randn(num_examples, in_dim)
Y_gt: np.ndarray = np.random.randn(num_examples, out_dim)


def test_layer_norm(X: np.ndarray,
                    Y_gt: np.ndarray) -> None:

    m: Sequential = Sequential()
    m.add(Dense(in_dim, 20))
    m.add(Tanh())

    m.add(LayerNorm())

    m.add(Dense(20, out_dim))
    m.add(Sigmoid())

    optim: SGD = SGD(m.parameters(), lr)
    loss_func: MSE = MSE()

    losses = list()
    for i in tqdm(list(range(max_epochs)), desc="checking gradients"):

        optim.reset()
        Y_hat = m.forward(X)
        losses.append(loss_func.forward(Y_hat, Y_gt))
        m.backward(X, loss_func.backward(Y_hat, Y_gt))

        grad_check(X, Y_gt, m, loss_func, delta=1e-5)
        optim.step()

    plt.plot(losses)
    plt.show()


def test_batch_norm(X: np.ndarray,
                    Y_gt: np.ndarray) -> None:

    m: Sequential = Sequential()
    m.add(Dense(in_dim, 20))
    m.add(Tanh())

    m.add(BatchNorm())

    m.add(Dense(20, out_dim))
    m.add(Sigmoid())

    optim: SGD = SGD(m.parameters(), lr)
    loss_func: MSE = MSE()

    losses = list()
    for i in tqdm(list(range(max_epochs)), desc="checking gradients"):

        optim.reset()
        Y_hat = m.forward(X)
        losses.append(loss_func.forward(Y_hat, Y_gt))
        m.backward(X, loss_func.backward(Y_hat, Y_gt))

        grad_check(X, Y_gt, m, loss_func, delta=1e-5)
        optim.step()

    plt.plot(losses)
    plt.show()


def main() -> None:
    # test_layer_norm(X, Y_gt)
    test_batch_norm(X, Y_gt)


if __name__ == "__main__":
    main()

