"""OpenAI-compatible chat model client."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from agentforge.models.base import BaseChatModel, ModelResponse


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
        if not normalized.startswith(("http://", "https://")):
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

    def _request_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"model": self.model, "messages": messages}

    def _build_url(self) -> str:
        parsed = urlparse(self.base_url)
        if self.force_chatcompletions_path:
            forced_path = self.force_chatcompletions_path
            if not forced_path.startswith("/"):
                forced_path = f"/{forced_path}"
            return urlunparse(parsed._replace(path=forced_path, params="", query="", fragment=""))
        path = parsed.path or ""
        if path in {"", "/"}:
            base_path = "/v1"
        else:
            base_path = path.rstrip("/")
            segments = [segment for segment in base_path.split("/") if segment]
            if "v1" not in segments:
                base_path = f"{base_path}/v1"
        if base_path.endswith("/chat/completions"):
            final_path = base_path
        else:
            final_path = f"{base_path}/chat/completions"
        return urlunparse(parsed._replace(path=final_path, params="", query="", fragment=""))

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ModelResponse:
        url = self._build_url()
        headers = {"Authorization": f"Bearer {self.api_key}", **self.extra_headers}
        payload = self._request_payload(messages)
        timeout = httpx.Timeout(self.timeout_seconds)

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=timeout, transport=self.transport) as client:
                    response = client.post(url, headers=headers, json=payload)
                if response.status_code in {429} or response.status_code >= 500:
                    raise OpenAICompatError(
                        f"Retryable error {response.status_code}: {response.text[:200]}"
                    )
                response.raise_for_status()
                if len(response.content) > self.max_response_bytes:
                    raise OpenAICompatError("Response too large")
                try:
                    data = response.json()
                except json.JSONDecodeError as exc:
                    raise OpenAICompatError("Malformed JSON response") from exc
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content")
                return ModelResponse(final_text=content if isinstance(content, str) else "")
            except (httpx.HTTPError, OpenAICompatError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(2**attempt)
        raise OpenAICompatError(f"OpenAI-compatible request failed: {last_error}")
