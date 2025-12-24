import os
import sys

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
