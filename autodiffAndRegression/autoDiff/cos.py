from .expr import Expression, ExpressionType
from .binop import BinaryOp, Op
from .const import Constant
import math

class Cos(Expression):
    def __init__(self, arg: Expression):
        self.arg = arg
    
    def __str__(self):
        return f"Cos({self.arg})"
    
    def __repr__(self):
        return self.__str__()
    
    def differentiate(self: ExpressionType) -> ExpressionType:
        from .sin import Sin
        diffArg = self.arg.differentiate()
        revDiffArg = BinaryOp(Constant(-1.), Op.MUL, diffArg)
        diffCos = Sin(self.arg)
        return BinaryOp(revDiffArg, Op.MUL, diffCos)
    
    def eval(self: ExpressionType, x: float) -> float:
        val = self.arg.eval(x)
        return math.cos(val)
    
    def deepcopy(self: ExpressionType) -> ExpressionType:
        c = Cos(self.arg.deepcopy())
        return c