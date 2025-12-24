"""Run a multi-file code project in a sandboxed workspace."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentforge.safety.sandbox import run_command
from agentforge.tools.base import Tool, ToolResult


class CodeRunMultiInput(BaseModel):
    files: dict[str, str]
    command: str
    timeout_seconds: int = Field(default=10, ge=1, le=30)


class CodeRunMultiTool(Tool):
    name = "code_run_multi"
    description = "Run a multi-file project in a temporary workspace directory."
    input_schema = CodeRunMultiInput

    def __init__(self, workspace_dir: str) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def run(self, data: BaseModel) -> ToolResult:
        payload = CodeRunMultiInput.model_validate(data)
        run_dir = self.workspace_dir / "code_runs" / uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)
        for rel_path, content in payload.files.items():
            target = (run_dir / rel_path).resolve()
            if not str(target).startswith(str(run_dir)):
                raise ValueError("File path escapes run directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        command_parts = shlex.split(payload.command)
        env = os.environ.copy()
        result = run_command(
            command_parts, cwd=run_dir, env=env, timeout_seconds=payload.timeout_seconds
        )
        return ToolResult(
            output={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "cwd": str(run_dir),
            }
        )
