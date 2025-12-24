from __future__ import annotations

import json

import httpx

from agentforge.models.openai_compat import OpenAICompatChatModel


def test_openai_base_url_and_tools_payload():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        data = json.loads(request.content.decode())
        assert "tools" in data
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "ok", "tool_calls": []}},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = OpenAICompatChatModel(
        base_url="https://example.com/v1/",
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )
    response = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
    )
    assert response.final_text == "ok"
    assert requests[0].url == httpx.URL("https://example.com/v1/chat/completions")
