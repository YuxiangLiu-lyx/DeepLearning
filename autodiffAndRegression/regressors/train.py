# SYSTEM IMPORTS
from typing import Type, Tuple
import pickle as pkl
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import os


np.random.seed(12345)


# PYTHON PROJECT IMPORTS
from models.lasso import LassoRegressor
from models.ridge import RidgeRegressor


# TYPES DECLARED IN THIS MODULE
AverageMeterType = Type["AverageMeter"]


# CONSTANTS
SALIENT = True
NUM_EPOCHS = 1000
LAMBDA_R = 1
LAMBDA_L = 0.5
BATCH_SIZE = 16
STEP_SIZE = 0.0001



class AverageMeter(object):
    def __init__(self: AverageMeterType) -> None:
        self.reset()

    def reset(self: AverageMeterType) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self: AverageMeterType,
               val: float,
               n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def permute_data(X: np.ndarray,
                 Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert(X.shape[0] == Y.shape[0])
    order = np.arange(X.shape[0])
    np.random.shuffle(order)
    return X[order, :], Y[order]


def train(X_train: np.ndarray,
          Y_train: np.ndarray,
          X_val: np.ndarray,
          Y_val: np.ndarray,
          m: object) -> None:
    num_samples = X_train.shape[0]

    for epoch in range(NUM_EPOCHS):
        epoch_loss = AverageMeter()

        X_t, Y_t = permute_data(X_train, Y_train)
        for batch_idx in range(0, num_samples, BATCH_SIZE):
            X_batch = X_t[batch_idx: min(batch_idx+BATCH_SIZE, num_samples)]
            Y_batch = Y_t[batch_idx: min(batch_idx+BATCH_SIZE, num_samples)]

            loss = m.loss(X_batch, Y_batch)
            epoch_loss.update(loss, n=X_batch.shape[0])

            grad = m.grad(X_batch, Y_batch)
            m.w -= STEP_SIZE * grad

        if not SALIENT:
            print(f"EPOCH [{epoch}/{NUM_EPOCHS}] (Avg. Loss: {epoch_loss.avg})")

    val_loss = m.loss(X_val, Y_val)
    if not SALIENT:
        print(f"EPOCH [{epoch}/{NUM_EPOCHS}] Validation Loss: {val_loss}")


def main() -> None:
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")
    data_file = os.path.join(data_dir, "data.pkl")

    data = None
    with open(data_file, "rb") as f:
        data = pkl.load(f)

    X_train = data["x_train"]
    Y_train = data["y_train"].reshape(-1, 1)
    X_val = data["x_test"]
    Y_val = data["y_test"].reshape(-1, 1)

    print(f"train shapes: {X_train.shape} -> {Y_train.shape}")
    print(f"val shapes: {X_val.shape} -> {Y_val.shape}")

    init_w: np.ndarray = np.random.randn(X_train.shape[1], 1)
    m_lasso: object = LassoRegressor(X_train.shape[1])
    m_lasso.w = init_w.copy()

    m_ridge: object = RidgeRegressor(X_train.shape[1])
    m_ridge.w = init_w.copy()

    train(X_train, Y_train, X_val, Y_val, m_lasso)
    train(X_train, Y_train, X_val, Y_val, m_ridge)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].stem(list(range(X_train.shape[1])), m_ridge.w)
    ax[0].grid()
    ax[0].set_title('Ridge Regression')
    ax[0].set_ylabel('w')


    ax[1].stem(list(range(X_train.shape[1])), m_lasso.w)
    ax[1].grid()
    ax[1].set_title('Lasso Regression')
    ax[1].set_ylabel('w')

    plt.show()


if __name__ == "__main__":
    main()


