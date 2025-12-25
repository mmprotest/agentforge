from agentforge.pre_solver import detect_pre_solvers


def test_math_queries_use_calculator_or_python():
    decisions = detect_pre_solvers("What is 2+2?")
    assert any(decision.tool_name == "calculator" for decision in decisions)


def test_unit_conversion_uses_deterministic_converter():
    decisions = detect_pre_solvers("convert 5 km to m")
    assert any(decision.tool_name == "unit_convert" for decision in decisions)


def test_regex_tasks_use_regex_tool():
    decisions = detect_pre_solvers('Extract /foo/ from "foo bar"')
    assert any(decision.tool_name == "regex_extract" for decision in decisions)
