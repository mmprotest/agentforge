"""PDF text extraction connector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentforge.tools.base import Tool, ToolResult, ToolError


class PdfExtractInput(BaseModel):
    path: str
    max_pages: int = Field(default=5)


class PdfExtractTool(Tool):
    name = "pdf_extract_text"
    description = "Extract text from PDF files in the workspace data directory."
    input_schema = PdfExtractInput

    def __init__(self, workspace_dir: str) -> None:
        self.root = Path(workspace_dir).resolve() / "data"
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if not str(target).startswith(str(self.root)):
            raise ValueError("Path traversal detected")
        return target

    def run(self, data: BaseModel | dict[str, Any]) -> ToolResult:
        payload = PdfExtractInput.model_validate(data)
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise ToolError(
                "Install agentforge[pdf] to enable PDF extraction."
            ) from exc
        target = self._safe_path(payload.path)
        reader = PdfReader(str(target))
        text_parts = []
        for page in reader.pages[: payload.max_pages]:
            text_parts.append(page.extract_text() or "")
        return ToolResult(output={"text": "\n".join(text_parts)})
