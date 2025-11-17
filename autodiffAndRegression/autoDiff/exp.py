from .expr import Expression, ExpressionType
from .binop import BinaryOp, Op
import math

class Exp(Expression):
    def __init__(self, arg: Expression):
        self.arg = arg
    
    def __str__(self):
        return f"(e^({self.arg}))"
    
    def __repr__(self):
        return self.__str__()
    
    def differentiate(self: ExpressionType) -> ExpressionType:
        diffArg = self.arg.differentiate()
        return BinaryOp(diffArg, Op.MUL, Exp(self.arg))
    
    def eval(self: ExpressionType, x: float) -> float:
        val = self.arg.eval(x)
        return math.exp(val)
    
    def deepcopy(self: ExpressionType) -> ExpressionType:
        c = Exp(self.arg.deepcopy())
        return c