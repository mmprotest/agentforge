import os
import sys

from agentforge.safety.sandbox import run_pytest
from agentforge.tools.builtins.code_run_multi import CodeRunMultiTool


def test_sandbox_env_filters_openai_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    tool = CodeRunMultiTool(str(tmp_path))
    result = tool.run(
        {
            "files": {
                "main.py": (
                    "import os\n"
                    "print(os.environ.get('OPENAI_API_KEY'))\n"
                )
            },
            "command": f"{sys.executable} main.py",
            "timeout_seconds": 5,
        }
    )
    stdout = (result.output.get("stdout") or "").strip()
    assert stdout in {"", "None"}


def test_run_pytest_sanitizes_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    test_file = tmp_path / "test_env_sanitized.py"
    test_file.write_text(
        "import os\n"
        "def test_env_sanitized():\n"
        "    assert os.environ.get('OPENAI_API_KEY') in {None, ''}\n"
    )
    result = run_pytest(test_file, timeout_seconds=10)
    assert result.success, result.output
