from __future__ import annotations

from pydantic import BaseModel

from agentforge.agent import Agent
from agentforge.models.base import ModelResponse, ToolCall
from agentforge.models.mock import MockChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.base import Tool, ToolResult
from agentforge.tools.registry import ToolRegistry


class DummyInput(BaseModel):
    value: int


class DummyTool(Tool):
    name = "dummy"
    description = "dummy tool"
    input_schema = DummyInput

    def run(self, data: BaseModel) -> ToolResult:
        payload = DummyInput.model_validate(data)
        return ToolResult(output={"ok": True, "value": payload.value})


def test_tool_vote_skips_invalid_candidate():
    scripted = [
        ModelResponse(tool_call=ToolCall(name="dummy", arguments={})),
        ModelResponse(tool_call=ToolCall(name="dummy", arguments={"value": 2})),
        ModelResponse(
            final_text='{"type":"final","answer":"done","confidence":0.2,"checks":[]}'
        ),
    ]
    model = MockChatModel(scripted=scripted)
    registry = ToolRegistry()
    registry.register(DummyTool())
    policy = SafetyPolicy(
        max_model_calls=5,
        tool_vote_k=1,
        tool_vote_max_samples=2,
        tool_vote_max_model_calls=2,
    )
    agent = Agent(model=model, registry=registry, policy=policy)
    result = agent.run("use dummy")
    assert "done" in result.answer
    assert "dummy" in result.tools_used


def test_red_flag_strict_json_discards_repaired_tool_calls():
    scripted = [
        ModelResponse(
            final_text='prefix {"type":"tool","name":"dummy","arguments":{"value":3}}'
        ),
        ModelResponse(tool_call=ToolCall(name="dummy", arguments={"value": 4})),
        ModelResponse(
            final_text='{"type":"final","answer":"ok","confidence":0.2,"checks":[]}'
        ),
    ]
    model = MockChatModel(scripted=scripted)
    registry = ToolRegistry()
    registry.register(DummyTool())
    policy = SafetyPolicy(
        max_model_calls=5,
        tool_vote_k=1,
        tool_vote_max_samples=2,
        tool_vote_max_model_calls=2,
        red_flag_strict_json=True,
    )
    agent = Agent(model=model, registry=registry, policy=policy)
    result = agent.run("use dummy")
    assert "ok" in result.answer
    assert "dummy" in result.tools_used


def test_tool_vote_stops_when_ahead_by_k():
    scripted = [
        ModelResponse(tool_call=ToolCall(name="dummy", arguments={"value": 1})),
        ModelResponse(tool_call=ToolCall(name="dummy", arguments={"value": 1})),
        ModelResponse(tool_call=ToolCall(name="dummy", arguments={"value": 2})),
    ]
    model = MockChatModel(scripted=scripted)
    registry = ToolRegistry()
    registry.register(DummyTool())
    policy = SafetyPolicy(
        max_model_calls=5,
        tool_vote_k=2,
        tool_vote_max_samples=5,
        tool_vote_max_model_calls=5,
    )
    agent = Agent(model=model, registry=registry, policy=policy)
    agent._messages = [{"role": "user", "content": "vote"}]
    tool_call = agent._elect_tool_call(registry.openai_schemas())
    assert tool_call is not None
    assert tool_call.name == "dummy"
    assert tool_call.arguments == {"value": 1}
    assert len(model._scripted) == 1
