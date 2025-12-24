"""OpenAI-compatible chat model client."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from agentforge.models.base import BaseChatModel, ModelResponse, ToolCall
from agentforge.util.logging import redact


class OpenAICompatError(RuntimeError):
    """Raised when the OpenAI-compatible backend returns an error."""


class OpenAICompatChatModel(BaseChatModel):
    """HTTP client for OpenAI-compatible chat/completions."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 30,
        max_response_bytes: int = 2_000_000,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.transport = transport

    def _parse_tool_call(self, message: dict[str, Any]) -> ToolCall | None:
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return None
        call = tool_calls[0]
        call_id = call.get("id")
        function = call.get("function") or {}
        name = function.get("name")
        raw_args = function.get("arguments") or "{}"
        arguments: dict[str, Any]
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {"raw": raw_args}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {"raw": raw_args}
        if not name:
            return None
        return ToolCall(id=call_id, name=name, arguments=arguments)

    def _request_payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ModelResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = self._request_payload(messages, tools)
        timeout = httpx.Timeout(self.timeout_seconds)

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=timeout, transport=self.transport) as client:
                    response = client.post(url, headers=headers, json=payload)
                if response.status_code in {429} or response.status_code >= 500:
                    raise OpenAICompatError(
                        f"Retryable error {response.status_code}: "
                        f"{redact(response.text[:200])}"
                    )
                response.raise_for_status()
                if len(response.content) > self.max_response_bytes:
                    raise OpenAICompatError("Response too large")
                data = response.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                tool_call = self._parse_tool_call(message)
                if tool_call:
                    return ModelResponse(tool_call=tool_call)
                return ModelResponse(final_text=message.get("content"))
            except (httpx.HTTPError, OpenAICompatError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(2**attempt)
        raise OpenAICompatError(f"OpenAI-compatible request failed: {last_error}")
