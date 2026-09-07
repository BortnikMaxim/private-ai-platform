"""Calculator tool: correctness and, mostly, refusal to do anything dangerous."""

import pytest

from backend.agent.tools.base import ToolContext, ToolError
from backend.agent.tools.calculator import (
    MAX_AST_DEPTH,
    MAX_EXPRESSION_LENGTH,
    CalculatorInput,
    CalculatorTool,
    calculate,
)

# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2+2", 4),
        ("2 + 2", 4),
        ("125 * 8", 1000),
        ("17 * 23", 391),
        ("15 * 1.2", 18.0),
        ("(2 + 3) * 4", 20),
        ("((1 + 2) * (3 + 4))", 21),
        ("-5 + 3", -2),
        ("-(4 + 6)", -10),
        ("+7", 7),
        ("2 ** 10", 1024),
        ("10 % 3", 1),
        ("7 / 2", 3.5),
        ("2 ** -2", 0.25),
    ],
)
def test_supported_arithmetic(expression, expected):
    assert calculate(expression) == pytest.approx(expected)


def test_unary_minus_binds_correctly():
    assert calculate("-2 ** 2") == pytest.approx(-4)
    assert calculate("(-2) ** 2") == pytest.approx(4)


# ---------------------------------------------------------------------------
# Controlled arithmetic failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expression", ["1/0", "10 / (5 - 5)", "5 % 0", "0 ** -1"])
def test_division_by_zero_is_a_controlled_error(expression):
    with pytest.raises(ToolError, match="zero"):
        calculate(expression)


def test_huge_exponent_is_rejected_without_being_computed():
    # Must fail on the bound, not by trying to build a 100M-digit number.
    with pytest.raises(ToolError, match="exponent"):
        calculate("10 ** 100000000")


def test_large_but_allowed_exponent_still_bounded_by_result_size():
    with pytest.raises(ToolError, match="too large"):
        calculate("1000 ** 30")


def test_result_magnitude_is_bounded():
    with pytest.raises(ToolError, match="too large"):
        calculate("99999999 * 99999999 * 99999999")


def test_expression_length_is_bounded():
    with pytest.raises(ToolError, match="too long"):
        calculate("1+" * (MAX_EXPRESSION_LENGTH // 2 + 5) + "1")


def _nested(levels: int) -> str:
    """'1+(1+(1+(...1...)))' — parentheses alone add no AST depth, operators do."""
    return "1+(" * levels + "1" + ")" * levels


def test_shallow_nesting_is_allowed():
    assert calculate(_nested(3)) == 4


def test_expression_depth_is_bounded():
    with pytest.raises(ToolError, match="too deeply"):
        calculate(_nested(MAX_AST_DEPTH + 5))


def test_a_flat_but_long_chain_hits_the_length_limit_not_a_crash():
    with pytest.raises(ToolError, match="too long"):
        calculate("+".join(["1"] * MAX_EXPRESSION_LENGTH))


# ---------------------------------------------------------------------------
# Security: nothing but arithmetic may pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "__import__('os')",
        "open('/etc/passwd').read()",
        "eval('2+2')",
        "exec('x=1')",
        "os.system('rm -rf /')",
        "().__class__.__bases__[0].__subclasses__()",
        "(lambda: 1)()",
        "[1,2,3]",
        "{'a': 1}",
        "x + 1",
        "abs(-3)",
        "1 if True else 2",
        "1 < 2",
        "1 and 2",
        "f'{1}'",
        "1;2",
        "import os",
        "globals()",
        "1 .__class__",
        "2 << 3",
        "5 & 3",
        "~5",
    ],
)
def test_non_arithmetic_expressions_are_refused(expression):
    with pytest.raises(ToolError):
        calculate(expression)


def test_no_builtin_is_reachable():
    for name in ("__import__", "eval", "exec", "open", "compile", "getattr"):
        with pytest.raises(ToolError):
            calculate(f"{name}('x')")


def test_boolean_literals_are_not_numbers():
    with pytest.raises(ToolError, match="numeric"):
        calculate("True + True")


def test_string_literals_are_rejected():
    with pytest.raises(ToolError):
        calculate("'a' * 1000000")


def test_malformed_expression_is_a_controlled_error():
    with pytest.raises(ToolError, match="could not be parsed"):
        calculate("2 +")


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------


async def test_tool_returns_integral_results_without_decimal_noise():
    tool = CalculatorTool()

    result = await tool.execute(CalculatorInput(expression="125 * 8"), ToolContext())

    assert result == {"expression": "125 * 8", "result": 1000}


async def test_tool_keeps_fractional_results():
    tool = CalculatorTool()

    result = await tool.execute(CalculatorInput(expression="7 / 2"), ToolContext())

    assert result["result"] == 3.5


def test_input_schema_rejects_an_overlong_expression():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CalculatorInput(expression="1" * (MAX_EXPRESSION_LENGTH + 1))
