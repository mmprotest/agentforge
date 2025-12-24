"""Python sandbox tool."""

from __future__ import annotations

import ast
import builtins
import multiprocessing
import os
import queue
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentforge.tools.base import Tool, ToolResult
from agentforge.safety.sandbox import sanitize_env


class PythonSandboxInput(BaseModel):
    code: str
    timeout_seconds: int = Field(default=2, ge=1, le=10)


class SandboxResult(BaseModel):
    result: Any | None
    error: str | None


class PythonSandboxTool(Tool):
    name = "python_sandbox"
    description = "Execute small Python snippets in a restricted sandbox."
    input_schema = PythonSandboxInput

    def __init__(self, workspace_dir: str) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def _validate_code(self, code: str) -> None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Imports are not allowed in sandbox")

    def _safe_open(self, path: str, mode: str = "r"):
        target = (self.workspace_dir / path).resolve()
        if not str(target).startswith(str(self.workspace_dir)):
            raise ValueError("Sandbox file access outside workspace")
        return open(target, mode, encoding="utf-8")

    def _run_code(self, code: str, output: multiprocessing.Queue) -> None:
        try:
            sanitized = sanitize_env(os.environ.copy())
            os.environ.clear()
            os.environ.update(sanitized)
            self._validate_code(code)
            safe_builtins = {
                "print": builtins.print,
                "len": builtins.len,
                "range": builtins.range,
                "sum": builtins.sum,
                "min": builtins.min,
                "max": builtins.max,
                "sorted": builtins.sorted,
                "open": self._safe_open,
            }
            globals_dict = {"__builtins__": safe_builtins}
            locals_dict: dict[str, Any] = {}
            exec(code, globals_dict, locals_dict)
            result = locals_dict.get("result")
            output.put({"result": result, "error": None})
        except Exception as exc:  # noqa: BLE001
            output.put({"result": None, "error": str(exc)})

    def run(self, data: BaseModel) -> ToolResult:
        input_data = PythonSandboxInput.model_validate(data)
        output_queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._run_code, args=(input_data.code, output_queue)
        )
        process.start()
        process.join(timeout=input_data.timeout_seconds)
        if process.is_alive():
            process.terminate()
            return ToolResult(output=SandboxResult(result=None, error="timeout").model_dump())
        try:
            result = output_queue.get_nowait()
        except queue.Empty:
            result = {"result": None, "error": "no output"}
        return ToolResult(output=SandboxResult(**result).model_dump())
