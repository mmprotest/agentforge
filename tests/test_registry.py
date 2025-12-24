from agentforge.tools.builtins.http_fetch import HttpFetchTool
from agentforge.tools.registry import ToolRegistry


def test_registry_register_and_list():
    registry = ToolRegistry()
    tool = HttpFetchTool()
    registry.register(tool)
    assert registry.get("http_fetch") is tool
    assert registry.list() == [tool]
    assert registry.openai_schemas()[0]["function"]["name"] == "http_fetch"
