"""OpenAI-compatible chat model client."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from agentforge.models.base import BaseChatModel, ModelResponse, ToolCall
from agentforge.protocol import parse_protocol, ProtocolToolCall
from agentforge.util.json_repair import JsonRepairError, repair_json
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
        extra_headers: dict[str, str] | None = None,
        disable_tool_choice: bool = False,
        force_chatcompletions_path: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized = base_url.strip()
        self._missing_scheme = not normalized.startswith(("http://", "https://"))
        if self._missing_scheme:
            normalized = f"http://{normalized}"
        self.base_url = normalized.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.extra_headers = extra_headers or {}
        self.disable_tool_choice = disable_tool_choice
        self.force_chatcompletions_path = force_chatcompletions_path
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
            if not self.disable_tool_choice:
                payload["tool_choice"] = "auto"
        return payload

    def _build_url(self) -> str:
        parsed = urlparse(self.base_url)
        if self.force_chatcompletions_path:
            forced_path = self.force_chatcompletions_path
            if not forced_path.startswith("/"):
                forced_path = f"/{forced_path}"
            return urlunparse(parsed._replace(path=forced_path, params="", query="", fragment=""))
        base_path = parsed.path.rstrip("/")
        if not base_path and self._missing_scheme:
            root_path = ""
        elif base_path.endswith("/v1"):
            root_path = base_path
        else:
            root_path = f"{base_path}/v1" if base_path else "/v1"
        final_path = f"{root_path}/chat/completions" if root_path else "/chat/completions"
        return urlunparse(parsed._replace(path=final_path, params="", query="", fragment=""))

    def _fallback_tool_call(self, content: str) -> ToolCall | None:
        protocol = parse_protocol(content)
        if isinstance(protocol, ProtocolToolCall):
            return ToolCall(id=None, name=protocol.name, arguments=protocol.arguments)
        try:
            payload = repair_json(content)
        except JsonRepairError:
            return None
        if not isinstance(payload, dict):
            return None
        name = payload.get("name") or payload.get("tool")
        arguments = payload.get("arguments") or {}
        if isinstance(name, str) and isinstance(arguments, dict):
            return ToolCall(id=None, name=name, arguments=arguments)
        return None

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ModelResponse:
        url = self._build_url()
        headers = {"Authorization": f"Bearer {self.api_key}", **self.extra_headers}
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
                try:
                    data = response.json()
                except json.JSONDecodeError as exc:
                    trimmed = redact(response.text[:200])
                    raise OpenAICompatError(f"Malformed JSON response: {trimmed}") from exc
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                tool_call = self._parse_tool_call(message)
                if tool_call:
                    return ModelResponse(tool_call=tool_call)
                content = message.get("content")
                if isinstance(content, str):
                    fallback = self._fallback_tool_call(content)
                    if fallback:
                        return ModelResponse(tool_call=fallback)
                return ModelResponse(final_text=content)
            except (httpx.HTTPError, OpenAICompatError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(2**attempt)
        raise OpenAICompatError(f"OpenAI-compatible request failed: {last_error}")
