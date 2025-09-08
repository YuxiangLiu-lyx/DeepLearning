from .expr import Expression, ExpressionType
from .const import Constant

class Variable(Expression):
    def __init__(self, var_x: str = "x"):
        self.var_x = var_x

    def __str__(self):
        return f"{self.var_x}"
    
    def __repr__(self):
        return self.__str__()
    
    def differentiate(self: ExpressionType) -> ExpressionType:
        return Constant(1)
    
    def eval(self: ExpressionType, x: float) -> float:
        return x
    
    def deepcopy(self: ExpressionType) -> ExpressionType:
        c = Variable(self.var_x)
        return c