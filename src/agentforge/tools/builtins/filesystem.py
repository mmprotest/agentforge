"""Filesystem tool restricted to workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentforge.tools.base import Tool, ToolResult


class FileSystemInput(BaseModel):
    action: str = Field(description="read|write|list")
    path: str
    content: str | None = None


class FileSystemTool(Tool):
    name = "filesystem"
    description = "Read/write/list files under the workspace directory."
    input_schema = FileSystemInput

    def __init__(self, workspace_dir: str) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, path: str) -> Path:
        target = (self.workspace_dir / path).resolve()
        if not str(target).startswith(str(self.workspace_dir)):
            raise ValueError("Path traversal detected")
        return target

    def run(self, data: BaseModel | dict[str, Any]) -> ToolResult:
        input_data = FileSystemInput.model_validate(data)
        target = self._safe_path(input_data.path)
        if input_data.action == "read":
            content = target.read_text(encoding="utf-8")
            return ToolResult(output={"content": content})
        if input_data.action == "write":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(input_data.content or "", encoding="utf-8")
            return ToolResult(output={"status": "ok"})
        if input_data.action == "list":
            if not target.exists():
                return ToolResult(output={"entries": []})
            entries = [path.name for path in target.iterdir()]
            return ToolResult(output={"entries": entries})
        raise ValueError("Unsupported filesystem action")
