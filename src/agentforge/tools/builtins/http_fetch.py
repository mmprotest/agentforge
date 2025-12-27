"""HTTP fetch tool."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from agentforge.tools.base import Tool, ToolResult


class HttpFetchInput(BaseModel):
    url: str
    timeout_seconds: int = Field(default=10, ge=1, le=60)
    max_bytes: int = Field(default=200_000, ge=1, le=1_000_000)
    headers: dict[str, str] | None = None


class HttpFetchOutput(BaseModel):
    status_code: int
    body: str


class HttpFetchTool(Tool):
    name = "http_fetch"
    description = "Fetch a URL via HTTP GET with limits."
    input_schema = HttpFetchInput
    output_schema = HttpFetchOutput

    def run(self, data: BaseModel | dict[str, Any]) -> ToolResult:
        input_data = HttpFetchInput.model_validate(data)
        with httpx.Client(timeout=input_data.timeout_seconds) as client:
            response = client.get(input_data.url, headers=input_data.headers)
        content = response.content[: input_data.max_bytes]
        output = HttpFetchOutput(status_code=response.status_code, body=content.decode("utf-8", errors="ignore"))
        return ToolResult(output=output.model_dump())
