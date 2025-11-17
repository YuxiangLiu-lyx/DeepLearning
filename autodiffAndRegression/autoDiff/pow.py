from .binop import BinaryOp, Op
from .expr import Expression, ExpressionType
from .const import Constant

class Power(Expression):
    def __init__(self, base: Expression, exp: float):
        self.base = base
        self.exp = float(exp)
        # self.exp = exp
    
    def __str__(self):
        return f"(({self.base})^{self.exp})"
    
    def __repr__(self):
        return self.__str__()
    
    def differentiate(self: ExpressionType) -> ExpressionType:
        diff1 = Power(self.base, self.exp - 1.)
        diffBase = self.base.differentiate()
        lhs = BinaryOp(Constant(self.exp), Op.MUL, diff1)
        return BinaryOp(lhs, Op.MUL, diffBase)
    
    def eval(self: ExpressionType, x: float) -> float:
        bs = self.base.eval(x)
        return pow(bs, self.exp)
    
    def deepcopy(self: ExpressionType) -> ExpressionType:
        c = Power(self.base, self.exp)
        return c