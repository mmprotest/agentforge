from agentforge.agent import Agent
from agentforge.models.base import ModelResponse
from agentforge.models.mock import MockChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.registry import ToolRegistry


def test_strict_json_mode_repairs_prefix():
    model = MockChatModel(
        scripted=[
            ModelResponse(
                final_text='here is JSON: {"type":"final","answer":"ok","confidence":0.4,"checks":[]}'
            )
        ]
    )
    agent = Agent(
        model=model,
        registry=ToolRegistry(),
        policy=SafetyPolicy(max_model_calls=2),
        strict_json_mode=True,
    )
    result = agent.run("hello")
    assert "ok" in result.answer


def test_strict_json_mode_format_retry_is_ephemeral():
    model = MockChatModel(
        scripted=[
            ModelResponse(final_text="not json at all"),
            ModelResponse(final_text='{"type":"final","answer":"ok","confidence":0.4,"checks":[]}'),
        ]
    )
    agent = Agent(
        model=model,
        registry=ToolRegistry(),
        policy=SafetyPolicy(max_model_calls=3),
        strict_json_mode=True,
    )
    result = agent.run("hello")
    assert "ok" in result.answer
    assert all(
        "Format error" not in message.get("content", "")
        for message in agent._messages
    )
