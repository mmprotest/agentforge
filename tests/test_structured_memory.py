from agentforge.agent import Agent
from agentforge.models.mock import MockChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.registry import ToolRegistry


def test_facts_structured_populated_from_tool_output():
    agent = Agent(
        model=MockChatModel(),
        registry=ToolRegistry(),
        policy=SafetyPolicy(),
        profile="agent",
    )
    agent._messages = []
    agent._handle_tool_result(
        "http_fetch",
        {"content": "See https://example.com/page"},
        call_id="http_fetch",
        arguments={"url": "https://example.com/page"},
    )
    structured = agent.memory.state.get("facts_structured", [])
    assert any(item.get("kind") == "url" for item in structured)
