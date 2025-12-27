from agentforge.tools.builtins.calculator import CalculatorTool


def test_calculator_exact_fraction():
    tool = CalculatorTool()
    result = tool.run({"expression": "1/3 + 1/6"})
    assert result.output["value"] == "1/2"
    assert result.output["rational"] == "1/2"


def test_calculator_caret_exponent_precedence():
    tool = CalculatorTool()
    result = tool.run({"expression": "2^3*4"})
    assert result.output["value"] == "32"
