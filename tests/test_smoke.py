from pathlib import Path

from agentforge.config import Settings
from agentforge.runtime.runtime import Runtime
from agentforge.runtime.smoke import run_smoke_test


def test_smoke_helper(tmp_path: Path) -> None:
    runtime = Runtime.from_workspace(tmp_path, "default")
    settings = Settings(workspace_dir=str(runtime.workspace.path))
    run_smoke_test(settings, runtime)
