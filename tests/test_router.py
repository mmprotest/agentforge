from agentforge.routing import suggest_tool


def test_router_detects_url():
    suggestion = suggest_tool("Fetch https://example.com")
    assert suggestion is not None
    assert suggestion.tool_name == "http_fetch"


def test_router_detects_arithmetic():
    suggestion = suggest_tool("What is 2+2?")
    assert suggestion is not None
    assert suggestion.tool_name == "calculator"
