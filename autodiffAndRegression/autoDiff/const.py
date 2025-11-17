from .expr import Expression, ExpressionType


class Constant(Expression):
    def __init__(self, val : float):
        self.val = val

    def __str__(self):
        return f"{self.val}"
    
    def __repr__(self):
        return self.__str__()
    
    def differentiate(self: ExpressionType) -> ExpressionType:
        return Constant(0.)
    
    def eval(self: ExpressionType, x: float) -> float:
        return self.val
    
    def deepcopy(self: ExpressionType) -> ExpressionType:
        c = Constant(self.val)
        return c