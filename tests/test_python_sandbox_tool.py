from agentforge.tools.builtins.python_sandbox import PythonSandboxTool


def test_python_sandbox_captures_last_expression(tmp_path):
    tool = PythonSandboxTool(str(tmp_path))
    result = tool.run({"code": "value = 2 + 2\nvalue", "timeout_seconds": 2})
    assert result.output["error"] is None
    assert result.output["result"] == 4
