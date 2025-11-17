from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from nn.module import Module
from nn.lf import LossFunction
from nn.layers.dense import Dense
from nn.layers.sigmoid import Sigmoid
from nn.layers.tanh import Tanh
from nn.layers.rnn import RNN
from nn.layers.cells.vanilla_rnn_cell import VanillaRNNCell
from nn.layers.timedistributed import TimeDistributed
from nn.models.sequential import Sequential
from nn.optimizers.sgd import SGDOptimizer as SGD
from nn.losses.mse import MeanSquaredError as MSE


from grad_check import grad_check


np.random.seed(12345)



lr: float = 0.1
max_epochs: int = 1000

batch_size = 2
seq_length = 5
num_features = 10
output_size = num_features
hidden_size = 10

X = np.zeros((batch_size, seq_length, num_features,))
Y_gt = np.zeros((batch_size, seq_length, num_features,))
for i in range(seq_length):
    X[0,i,i] = 1
    Y_gt[0,i,(i+1)%num_features] = 1
for i in range(seq_length):
    X[1,i,i+5]=1
    Y_gt[1,i,(i+5+1)%num_features] = 1


m: Sequential = Sequential()
m.add(TimeDistributed(Dense(num_features, num_features)))
m.add(RNN(VanillaRNNCell(num_features, hidden_size, output_size,
                         output_activation=Sigmoid()),
          return_sequences=True))
m.add(TimeDistributed(Dense(output_size, output_size)))

optim: SGD = SGD(m.parameters(), lr)
loss_func: LossFunction = MSE()

losses = list()
for i in tqdm(list(range(max_epochs)), desc="checking gradients"):

    optim.reset()
    Y_hat = m.forward(X)
    losses.append(loss_func.forward(Y_hat, Y_gt))
    m.backward(X, loss_func.backward(Y_hat, Y_gt))

    grad_check(X, Y_gt, m, loss_func, epsilon=1e-5)
    optim.step()

plt.plot(losses)
plt.show()

