"""Arithmetic evaluation with an AST allowlist.

No ``eval``, no ``exec``, no shell, no imports, no attribute access, no names.
The expression is parsed with :mod:`ast` and walked by hand; anything that is
not one of the whitelisted node types is rejected before evaluation. Size,
depth and magnitude are all bounded, so a hostile expression cannot burn CPU
either.
"""

import ast
import math
from typing import Any

from pydantic import BaseModel, Field

from backend.agent.tools.base import Tool, ToolContext, ToolError

MAX_EXPRESSION_LENGTH = 200
MAX_AST_DEPTH = 12
MAX_ABS_VALUE = 1e15
MAX_EXPONENT = 32
# Reject a power before computing it if the result would obviously blow up.
MAX_RESULT_DIGITS = 15

# The complete allowlist. Anything absent is a hard failure.
ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
)

ALLOWED_OPERATORS: tuple[type[ast.AST], ...] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.UAdd,
    ast.USub,
)


class CalculatorInput(BaseModel):
    expression: str = Field(
        min_length=1,
        max_length=MAX_EXPRESSION_LENGTH,
        description="Arithmetic expression, e.g. '15 * 1.2' or '(2 + 3) ** 2'",
    )


def _check_value(value: float) -> float:
    if isinstance(value, complex):
        raise ToolError("complex results are not supported")

    if math.isnan(value) or math.isinf(value):
        raise ToolError("result is not a finite number")

    if abs(value) > MAX_ABS_VALUE:
        raise ToolError("result is too large")

    return value


def _evaluate(node: ast.AST, depth: int = 0) -> float:
    if depth > MAX_AST_DEPTH:
        raise ToolError("expression is nested too deeply")

    if isinstance(node, ast.Expression):
        return _evaluate(node.body, depth + 1)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ToolError("only numeric literals are allowed")
        return _check_value(float(node.value))

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ALLOWED_OPERATORS):
            raise ToolError(f"unsupported unary operator: {type(node.op).__name__}")

        operand = _evaluate(node.operand, depth + 1)
        return _check_value(-operand if isinstance(node.op, ast.USub) else operand)

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ALLOWED_OPERATORS):
            raise ToolError(f"unsupported operator: {type(node.op).__name__}")

        left = _evaluate(node.left, depth + 1)
        right = _evaluate(node.right, depth + 1)

        return _check_value(_apply(node.op, left, right))

    raise ToolError(f"unsupported expression element: {type(node).__name__}")


def _apply(op: ast.AST, left: float, right: float) -> float:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div):
        if right == 0:
            raise ToolError("division by zero")
        return left / right
    if isinstance(op, ast.Mod):
        if right == 0:
            raise ToolError("modulo by zero")
        return left % right
    if isinstance(op, ast.Pow):
        return _power(left, right)

    raise ToolError(f"unsupported operator: {type(op).__name__}")


def _power(base: float, exponent: float) -> float:
    if abs(exponent) > MAX_EXPONENT:
        raise ToolError(f"exponent must be between -{MAX_EXPONENT} and {MAX_EXPONENT}")

    # Estimate the magnitude *before* computing, so 10 ** 100000000 is rejected
    # rather than evaluated.
    if base != 0:
        digits = abs(exponent) * math.log10(abs(base))
        if digits > MAX_RESULT_DIGITS:
            raise ToolError("result is too large")

    if base == 0 and exponent < 0:
        raise ToolError("division by zero")

    try:
        return float(base**exponent)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ToolError("expression could not be evaluated") from exc


ALLOWED_ALL: tuple[type[ast.AST], ...] = ALLOWED_NODES + ALLOWED_OPERATORS


def _validate_nodes(tree: ast.AST) -> None:
    """Reject the whole tree up front; only then is evaluation attempted."""
    for node in ast.walk(tree):
        if isinstance(node, ALLOWED_ALL):
            continue
        raise ToolError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> float:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError("expression is too long")

    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise ToolError("expression could not be parsed") from exc

    # Reject the whole tree up front, then evaluate the vetted nodes.
    _validate_nodes(tree)

    return _evaluate(tree)


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Вычисляет арифметическое выражение: + - * / ** % и скобки. "
        "Только числа, без переменных и функций."
    )
    input_schema = CalculatorInput

    async def execute(
        self,
        arguments: CalculatorInput,
        context: ToolContext,
    ) -> dict[str, Any]:
        value = calculate(arguments.expression)

        # Present integral results without a trailing .0.
        rendered = int(value) if float(value).is_integer() else round(value, 10)

        return {"expression": arguments.expression, "result": rendered}
