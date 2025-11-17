# SYSTEM IMPORTS
from typing import Type
import numpy as np


# PYTHON PROJECT IMPORTS


# TYPES DECLARED IN THIS MODULE
RidgeRegressorType = Type["RidgeRegressorType"]


# CONSTANTS



class RidgeRegressor(object):
    def __init__(self: RidgeRegressorType,
                 num_features: int,
                 regularizer_coeff: float = 1) -> None:
        self.w: np.ndarray = np.random.randn(num_features, 1)
        self.regularizer_coeff: float = regularizer_coeff

    def predict(self: RidgeRegressorType,
                X: np.ndarray) -> np.ndarray:
        """
            A method to calculate predictions using ridge regression.
            @param X: the matrix of input examples. Has shape (num_examples, num_features))
            @return np.ndarray: the matrix of predictions. Has shape (num_examples, 1)
        """
        return np.dot(X, self.w)

    def loss(self: RidgeRegressorType,
             Y_hat: np.ndarray,
             Y_gt: np.ndarray) -> float:
        """
            A method to calculate the loss function for ridge regression.
            @param Y_hat: the matrix of predictions. Has shape (num_examples, 1)
            @param Y_gt: the matrix of ground truth. Has shape (num_examples, 1)
            @return float: the ridge regression loss function evaluated at Y_hat, Y_gt
        """
        mse = np.sum((Y_hat - Y_gt) ** 2)
        l2_regularization = self.regularizer_coeff * np.sum(self.w ** 2)
        return mse + l2_regularization

    def grad(self: RidgeRegressorType,
             X: np.ndarray,
             Y_gt: np.ndarray) -> np.ndarray:
        """
            A method to calculate the gradient of the ridge loss function with respect to the parameters 'self.w'
            @param X: the matrix of input examples. Has shape (num_examples, num_features)
            @param Y_gt: the matrix of ground truth. Has shape (num_features, 1)
            @return np.ndarray: the gradient of the ridge loss function with respect to 'self.w'. Has shape (num_features, 1)
        """
        return 2. * X.T @ (self.predict(X) - Y_gt) + 2 * self.regularizer_coeff * self.w

