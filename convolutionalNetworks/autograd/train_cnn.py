# SYSTEM IMPORTS
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


_CD_: str = os.path.abspath(os.path.dirname(__file__))
if _CD_ not in sys.path:
    sys.path.append(_CD_)
del _CD_


# PYTHON PROJECT IMPORTS
from nn.module import Module
from nn.param import Parameter
from nn.lf import LossFunction
from nn.layers.conv2d import Conv2d
from nn.layers.dense import Dense
from nn.layers.flatten import Flatten
from nn.layers.maxpool2d import MaxPool2d
from nn.layers.relu import ReLU
from nn.layers.softmax import Softmax
from nn.models.sequential import Sequential
from nn.optimizers.sgd import SGDOptimizer as SGD
from nn.losses.cross_entropy import CategoricalCrossEntropy
from grad_check import grad_check


np.random.seed(12345)


num_examples: int =  200
num_classes: int = 3

img_height: int = 10
img_width: int = 10
num_channels: int = 3

kernel_size: int = 3
num_kernels: int = 4

pool_size: Tuple[int, int] = (2, 2)

lr: float = 0.1
max_epochs: int = 100

X: np.ndarray = np.random.randn(num_examples, img_height, img_width, num_channels)

classes: np.ndarray = np.random.randint(num_classes, size=num_examples)
Y_gt: np.ndarray = np.zeros((num_examples, num_classes), dtype=float)
Y_gt[np.arange(num_examples), classes] = 1


m: Sequential = Sequential()
m.add(Conv2d(num_kernels=num_kernels,
             num_channels=num_channels,
             kernel_size=kernel_size))
m.add(ReLU())
m.add(MaxPool2d(pool_size=pool_size))
m.add(Flatten())
m.add(Dense(64, num_classes))
m.add(Softmax())

optim: SGD = SGD(m.parameters(), lr)
loss_func: LossFunction = CategoricalCrossEntropy()

losses = list()
for i in tqdm(list(range(max_epochs)), desc="checking gradients"):

    optim.reset()
    Y_hat = m.forward(X)
    losses.append(loss_func.forward(Y_hat, Y_gt))
    m.backward(X, loss_func.backward(Y_hat, Y_gt))

    grad_check(X, Y_gt, m, loss_func, epsilon=1e-6)
    optim.step()


plt.plot(losses)
plt.show()

