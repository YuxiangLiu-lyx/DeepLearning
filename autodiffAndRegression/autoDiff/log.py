from .expr import Expression, ExpressionType
from .binop import BinaryOp, Op
import math

class Log(Expression):
    def __init__(self, arg: Expression):
        self.arg = arg
    
    def __str__(self):
        return f"(log({self.arg}))"
    
    def __repr__(self):
        return self.__str__()
    
    def differentiate(self: ExpressionType) -> ExpressionType:
        diffArg = self.arg.differentiate()
        return BinaryOp(diffArg, Op.DIV, self.arg)

    def eval(self: ExpressionType, x: float) -> float:
        val = self.arg.eval(x)
        if val <= 0:
            raise ValueError("ERROR: Logarithm of non-positive value")
        return math.log(val)
    
    def deepcopy(self: ExpressionType) -> ExpressionType:
        c = Log(self.arg.deepcopy())
        return c