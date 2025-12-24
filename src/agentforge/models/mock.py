"""Mock chat model for offline testing."""

from __future__ import annotations

import json
from typing import Any

from agentforge.models.base import BaseChatModel, ModelResponse, ToolCall


class MockChatModel(BaseChatModel):
    """Deterministic mock model used when no API key is available."""

    def __init__(self, scripted: list[ModelResponse] | None = None) -> None:
        self._scripted = scripted or []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ModelResponse:
        if self._scripted:
            return self._scripted.pop(0)
        last = messages[-1].get("content") if messages else ""
        if isinstance(last, str) and last.startswith("USE_TOOL:"):
            response = self._tool_call_from_prompt(last, tools)
            if response is not None:
                return response
        return ModelResponse(final_text=f"Mock response to: {last}")

    def _tool_call_from_prompt(
        self, prompt: str, tools: list[dict[str, Any]] | None
    ) -> ModelResponse | None:
        if not tools:
            return ModelResponse(final_text=f"Mock response to: {prompt} (error: no tools available)")
        marker = "USE_TOOL:"
        stripped = prompt[len(marker) :].strip()
        if not stripped:
            return ModelResponse(final_text=f"Mock response to: {prompt} (error: missing tool name)")
        parts = stripped.split(maxsplit=1)
        tool_name = parts[0]
        tool_args = parts[1] if len(parts) > 1 else "{}"
        tool_names = {tool["function"]["name"] for tool in tools}
        if tool_name not in tool_names:
            return ModelResponse(final_text=f"Mock response to: {prompt} (error: unknown tool)")
        try:
            arguments = json.loads(tool_args)
        except json.JSONDecodeError:
            return ModelResponse(final_text=f"Mock response to: {prompt} (error: invalid JSON args)")
        if not isinstance(arguments, dict):
            return ModelResponse(final_text=f"Mock response to: {prompt} (error: args must be object)")
        return ModelResponse(tool_call=ToolCall(name=tool_name, arguments=arguments))
