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


def _has_result_assignment(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets):
                return True
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "result":
                return True
        if isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name) and node.target.id == "result":
                return True
    return False


def _prepare_code(code: str) -> ast.AST:
    tree = ast.parse(code)
    if not tree.body:
        return tree
    if _has_result_assignment(tree):
        return tree
    last_statement = tree.body[-1]
    if isinstance(last_statement, ast.Expr):
        assign = ast.Assign(
            targets=[ast.Name(id="result", ctx=ast.Store())],
            value=last_statement.value,
        )
        tree.body[-1] = assign
        ast.fix_missing_locations(tree)
    return tree


def _run_sandboxed_code(code: str, workspace_dir: str, output: multiprocessing.Queue) -> None:
    try:
        sanitized = sanitize_env(os.environ.copy())
        os.environ.clear()
        os.environ.update(sanitized)
        safe_root = Path(workspace_dir).resolve()

        def safe_open(path: str, mode: str = "r"):
            target = (safe_root / path).resolve()
            if not str(target).startswith(str(safe_root)):
                raise ValueError("Sandbox file access outside workspace")
            return open(target, mode, encoding="utf-8")

        compiled = compile(_prepare_code(code), "<sandbox>", "exec")
        safe_builtins = {
            "print": builtins.print,
            "len": builtins.len,
            "range": builtins.range,
            "sum": builtins.sum,
            "min": builtins.min,
            "max": builtins.max,
            "sorted": builtins.sorted,
            "open": safe_open,
            "__import__": builtins.__import__,
        }
        globals_dict = {"__builtins__": safe_builtins}
        locals_dict: dict[str, Any] = {}
        exec(compiled, globals_dict, locals_dict)
        result = locals_dict.get("result")
        output.put({"result": result, "error": None})
    except Exception as exc:  # noqa: BLE001
        output.put({"result": None, "error": str(exc)})


class PythonSandboxTool(Tool):
    name = "python_sandbox"
    description = "Execute small Python snippets in a restricted sandbox."
    input_schema = PythonSandboxInput

    def __init__(self, workspace_dir: str) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def _validate_code(self, code: str) -> None:
        ast.parse(code)

    def run(self, data: BaseModel | dict[str, Any]) -> ToolResult:
        input_data = PythonSandboxInput.model_validate(data)
        output_queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_run_sandboxed_code,
            args=(input_data.code, str(self.workspace_dir), output_queue),
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
