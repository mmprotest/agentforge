from __future__ import annotations

import json

import pytest

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


def test_openai_extra_headers_and_disable_tool_choice():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        data = json.loads(request.content.decode())
        assert "tool_choice" not in data
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    client = OpenAICompatChatModel(
        base_url="https://example.com",
        api_key="test-key",
        model="gpt-test",
        extra_headers={"X-Test": "yes"},
        disable_tool_choice=True,
        transport=transport,
    )
    response = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
    )
    assert response.final_text == "ok"
    assert requests[0].headers["X-Test"] == "yes"


def test_openai_tool_call_fallback_from_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"type":"tool","name":"calculator","arguments":{"expression":"2+2"}}'
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = OpenAICompatChatModel(
        base_url="https://example.com/v1",
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )
    response = client.chat(messages=[{"role": "user", "content": "hi"}], tools=None)
    assert response.tool_call is not None
    assert response.tool_call.name == "calculator"


def test_openai_base_url_without_v1():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatChatModel(
        base_url="https://example.com",
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )
    client.chat(messages=[{"role": "user", "content": "hi"}], tools=None)
    assert requests[0].url == httpx.URL("https://example.com/v1/chat/completions")


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://host:8000", "http://host:8000/v1/chat/completions"),
        ("http://host:8000/v1", "http://host:8000/v1/chat/completions"),
        ("http://host:8000/api/v1", "http://host:8000/api/v1/chat/completions"),
        ("http://host:8000/api/v1/", "http://host:8000/api/v1/chat/completions"),
    ],
)
def test_openai_base_url_variants(base_url: str, expected: str):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatChatModel(
        base_url=base_url,
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )
    client.chat(messages=[{"role": "user", "content": "hi"}], tools=None)
    assert requests[0].url == httpx.URL(expected)


def test_openai_force_chatcompletions_path():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatChatModel(
        base_url="http://host:8000/api",
        api_key="test-key",
        model="gpt-test",
        force_chatcompletions_path="/v1/chat/completions",
        transport=transport,
    )
    client.chat(messages=[{"role": "user", "content": "hi"}], tools=None)
    assert requests[0].url == httpx.URL("http://host:8000/v1/chat/completions")
