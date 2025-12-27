"""JSON repair tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentforge.tools.base import Tool, ToolResult
from agentforge.util.json_repair import JsonRepairError, repair_json


class JsonRepairInput(BaseModel):
    text: str


class JsonRepairTool(Tool):
    name = "json_repair"
    description = "Repair and parse JSON-like strings."
    input_schema = JsonRepairInput

    def run(self, data: BaseModel | dict[str, Any]) -> ToolResult:
        payload = JsonRepairInput.model_validate(data)
        try:
            repaired = repair_json(payload.text)
            return ToolResult(output={"value": repaired})
        except JsonRepairError as exc:
            return ToolResult(output={"error": str(exc), "value": None})
