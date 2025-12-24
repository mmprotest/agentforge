"""Mock chat model for offline testing."""

from __future__ import annotations

from typing import Any

from agentforge.models.base import BaseChatModel, ModelResponse, ToolCall


class MockChatModel(BaseChatModel):
    """Deterministic mock model used when no API key is available."""

    def __init__(self, scripted: list[ModelResponse] | None = None) -> None:
        self._scripted = scripted or []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ModelResponse:
        if self._scripted:
            return self._scripted.pop(0)
        if tools:
            first_tool = tools[0]["function"]["name"]
            return ModelResponse(tool_call=ToolCall(name=first_tool, arguments={}))
        last = messages[-1].get("content") if messages else ""
        return ModelResponse(final_text=f"Mock response to: {last}")
