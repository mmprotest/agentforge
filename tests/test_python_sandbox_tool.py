from agentforge.tools.builtins.python_sandbox import PythonSandboxTool


def test_python_sandbox_captures_last_expression(tmp_path):
    tool = PythonSandboxTool(str(tmp_path))
    result = tool.run({"code": "value = 2 + 2\nvalue", "timeout_seconds": 2})
    assert result.output["error"] is None
    assert result.output["result"] == 4


def test_python_sandbox_allows_math_import(tmp_path):
    tool = PythonSandboxTool(str(tmp_path))
    result = tool.run({"code": "import math\nmath.sqrt(9)", "timeout_seconds": 2})
    assert result.output["error"] is None
    assert result.output["result"] == 3.0


def test_python_sandbox_blocks_os_import(tmp_path):
    tool = PythonSandboxTool(str(tmp_path))
    result = tool.run({"code": "import os\nos.listdir('.')", "timeout_seconds": 2})
    assert result.output["error"] is not None
    assert "Import blocked" in result.output["error"]
