"""Base tool definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    output: Any


class Tool(ABC):
    """Abstract tool."""

    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel] | None = None

    @abstractmethod
    def run(self, data: BaseModel | dict[str, Any]) -> ToolResult:
        """Execute the tool."""
        raise NotImplementedError

    def openai_schema(self) -> dict[str, Any]:
        """Return OpenAI-compatible tool schema."""
        schema = self.input_schema.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }
