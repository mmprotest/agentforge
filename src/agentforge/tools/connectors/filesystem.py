"""Filesystem connector tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentforge.tools.base import Tool, ToolResult


class FilesystemConnectorInput(BaseModel):
    action: str = Field(description="read|list")
    path: str
    max_bytes: int = Field(default=20000)


class FilesystemConnectorTool(Tool):
    name = "filesystem_connector"
    description = "Read-only filesystem connector scoped to workspace data directory."
    input_schema = FilesystemConnectorInput

    def __init__(self, workspace_dir: str, read_only: bool = True) -> None:
        self.root = Path(workspace_dir).resolve() / "data"
        self.root.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only

    def _safe_path(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if not str(target).startswith(str(self.root)):
            raise ValueError("Path traversal detected")
        return target

    def run(self, data: BaseModel | dict[str, Any]) -> ToolResult:
        payload = FilesystemConnectorInput.model_validate(data)
        target = self._safe_path(payload.path)
        if payload.action == "read":
            content = target.read_bytes()[: payload.max_bytes]
            return ToolResult(output={"content": content.decode("utf-8", errors="replace")})
        if payload.action == "list":
            if not target.exists():
                return ToolResult(output={"entries": []})
            entries = [path.name for path in target.iterdir()]
            return ToolResult(output={"entries": entries})
        raise ValueError("Unsupported filesystem connector action")
