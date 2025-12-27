"""Calculator tool for exact arithmetic."""

from __future__ import annotations

import ast
from fractions import Fraction

from pydantic import BaseModel

from agentforge.tools.base import Tool, ToolResult


class CalculatorInput(BaseModel):
    expression: str


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate arithmetic expressions with exact fractions."
    input_schema = CalculatorInput

    def run(self, data: BaseModel) -> ToolResult:
        payload = CalculatorInput.model_validate(data)
        value = _evaluate_expression(payload.expression)
        value_text = str(value.numerator // value.denominator) if value.denominator == 1 else str(value)
        return ToolResult(
            output={
                "value": value_text,
                "rational": f"{value.numerator}/{value.denominator}",
            }
        )


def _evaluate_expression(expression: str) -> Fraction:
    normalized = expression.replace("^", "**")
    tree = ast.parse(normalized, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_node(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _apply_operator(node.op, left, right)
    raise ValueError("Unsupported expression")


def _apply_operator(op: ast.AST, left: Fraction, right: Fraction) -> Fraction:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div):
        return left / right
    if isinstance(op, ast.FloorDiv):
        return left // right
    if isinstance(op, ast.Mod):
        return left % right
    if isinstance(op, ast.Pow):
        if right.denominator != 1:
            raise ValueError("Exponent must be integer")
        return left**int(right)
    if isinstance(op, ast.BitXor):
        if right.denominator != 1:
            raise ValueError("Exponent must be integer")
        return left**int(right)
    raise ValueError("Unsupported operator")
