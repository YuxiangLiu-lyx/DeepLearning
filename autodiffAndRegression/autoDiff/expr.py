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

class Constant(Expression):
    def __init__(self, val : float):
        self.val = val

    def __str__(self):
        return f"{self.val}"
    
    def __repr__(self, val):
        return f"Constant(val={self.val})"
    
    def differentiate(self: ExpressionType) -> ExpressionType:
        return Constant(0.)
    
    def eval(self: ExpressionType,
             x: float) -> float:
        return self.val
    
    def deepcopy(self: ExpressionType) -> ExpressionType:
        c = Constant(self.val)
        return c
