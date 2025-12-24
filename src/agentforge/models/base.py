"""Base model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str | None = None
    name: str
    arguments: dict[str, Any]


class ModelResponse(BaseModel):
    final_text: str | None = None
    tool_call: ToolCall | None = None


class BaseChatModel(ABC):
    """Abstract chat model interface."""

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ModelResponse:
        """Send chat request and return model response."""
        raise NotImplementedError
