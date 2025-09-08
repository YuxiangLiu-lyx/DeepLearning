from enum import Enum
from typing import Type
from .expr import Expression, ExpressionType

OpType = Type["Op"]

class Op(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4

    def __str__(self: OpType) -> str:
        op_str: str = None
        if self == Op.ADD:
            op_str = "+"
        elif self == Op.SUB:
            op_str = "-"
        elif self == Op.MUL:
            op_str = "*"
        elif self == Op.DIV:
            op_str = "/"
        else:
            raise ValueError("ERROR: unknown op [{0}]".format(self))
        return op_str

    def __repr__(self: OpType) -> str:
        return self.__str__()
    
class BinaryOp(Expression):
    def __init__(self, lhs: Expression, op: Op, rhs: Expression):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def __str__(self):
        return f"({self.lhs}{self.op}{self.rhs})"
    
    def __repr__(self):
        return self.__str__()
    
    def differentiate(self: ExpressionType) -> ExpressionType:
        from .pow import Power
        left_diff = self.lhs.differentiate()
        right_diff = self.rhs.differentiate()
        if self.op == Op.ADD:
            return BinaryOp(left_diff, Op.ADD, right_diff)
        elif self.op == Op.SUB:
            return BinaryOp(left_diff, Op.SUB, right_diff)
        elif self.op == Op.MUL:
            diff1 = BinaryOp(left_diff, Op.MUL, self.rhs)
            diff2 = BinaryOp(self.lhs, Op.MUL, right_diff)
            return BinaryOp(diff1, Op.ADD, diff2)
        elif self.op == Op.DIV:
            diff1 = BinaryOp(self.rhs, Op.MUL, left_diff)
            diff2 = BinaryOp(self.lhs, Op.MUL, right_diff)
            sub = BinaryOp(diff1, Op.SUB, diff2)
            mul = Power(self.rhs, 2)
            return BinaryOp(sub, Op.DIV, mul)
        else:
            raise ValueError(f"ERROR: Unknown operator {self.op}")
    
    def eval(self: ExpressionType, x: float) -> float:
        left_value = self.lhs.eval(x)
        right_value = self.rhs.eval(x)
        if self.op == Op.ADD:
            return left_value + right_value
        elif self.op == Op.SUB:
            return left_value - right_value
        elif self.op == Op.MUL:
            return left_value * right_value
        elif self.op == Op.DIV:
            if right_value == 0:
                raise ValueError("ERROR: Division by zero")
            return left_value / right_value
        else:
            raise ValueError(f"ERROR: Unknown operator {self.op}")
    
    def deepcopy(self: ExpressionType) -> ExpressionType:
        c = BinaryOp(self.lhs.deepcopy(), self.op, self.rhs.deepcopy())
        return c
