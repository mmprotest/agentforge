from agentforge.tools.builtins.code_run_multi import CodeRunMultiTool


def test_code_run_multi_executes(tmp_path):
    tool = CodeRunMultiTool(str(tmp_path))
    result = tool.run(
        {"files": {"main.py": 'print("hello")'}, "command": "python main.py"}
    )
    assert result.output["exit_code"] == 0
    assert "hello" in result.output["stdout"]
