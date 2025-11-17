# SYSTEM IMPORTS
from typing import Type
from abc import ABC, abstractmethod


# PYTHON PROJECT IMPORTS


# TYPES DECLARED IN THIS MODULE
ExpressionType = Type["Expression"]


# CONSTANTS


class Expression(ABC):
    def __init__(self: ExpressionType) -> None:
        ...

    @abstractmethod
    def differentiate(self: ExpressionType) -> ExpressionType:
        ...

    @abstractmethod
    def eval(self: ExpressionType,
             x: float) -> float:
        ...

    @abstractmethod
    def deepcopy(self: ExpressionType) -> ExpressionType:
        ...

