from .expr import Expression, ExpressionType
from .binop import BinaryOp, Op
import math

class Sin(Expression):
    def __init__(self, arg: Expression):
        self.arg = arg
    
    def __str__(self):
        return f"Sin({self.arg})"
    
    def __repr__(self):
        return self.__str__()
    
    def differentiate(self: ExpressionType) -> ExpressionType:
        from .cos import Cos
        diffArg = self.arg.differentiate()
        diffSin = Cos(self.arg)
        return BinaryOp(diffArg, Op.MUL, diffSin)
    
    def eval(self: ExpressionType, x: float) -> float:
        val = self.arg.eval(x)
        return math.sin(val)
    
    def deepcopy(self: ExpressionType) -> ExpressionType:
        c = Sin(self.arg.deepcopy())
        return c