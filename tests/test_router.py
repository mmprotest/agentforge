from agentforge.routing import suggest_tool


def test_router_detects_url():
    suggestion = suggest_tool("Fetch https://example.com")
    assert suggestion is not None
    assert suggestion.tool_name == "http_fetch"


def test_router_detects_arithmetic():
    suggestion = suggest_tool("What is 2+2?")
    assert suggestion is not None
    assert suggestion.tool_name == "calculator"


def test_router_detects_unit_convert():
    suggestion = suggest_tool("Convert 10km to mi")
    assert suggestion is not None
    assert suggestion.tool_name == "unit_convert"


def test_router_ignores_generic_to_phrase():
    suggestion = suggest_tool("Please send this to the manager")
    assert suggestion is None
