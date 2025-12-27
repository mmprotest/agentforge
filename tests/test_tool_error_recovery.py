from __future__ import annotations

from agentforge.agent import Agent
from agentforge.models.base import ModelResponse, ToolCall
from agentforge.models.mock import MockChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.base import Tool, ToolResult
from agentforge.tools.registry import ToolRegistry
from agentforge.trace import TraceRecorder
from pydantic import BaseModel


class DummyInput(BaseModel):
    value: int


class DummyTool(Tool):
    name = "dummy"
    description = "dummy tool"
    input_schema = DummyInput

    def run(self, data: BaseModel) -> ToolResult:
        payload = DummyInput.model_validate(data)
        return ToolResult(output={"ok": True, "value": payload.value})


def test_tool_error_recovery_returns_final(tmp_path):
    scripted = [
        ModelResponse(tool_call=ToolCall(name="dummy", arguments={})),
        ModelResponse(tool_call=ToolCall(name="dummy", arguments={"value": 3})),
        ModelResponse(
            final_text='{"type":"final","answer":"done","confidence":0.2,"checks":[]}'
        ),
    ]
    model = MockChatModel(scripted=scripted)
    registry = ToolRegistry()
    registry.register(DummyTool())
    trace = TraceRecorder(trace_id="tool-error", workspace_dir=str(tmp_path))
    agent = Agent(
        model=model,
        registry=registry,
        policy=SafetyPolicy(
            max_model_calls=5,
            tool_vote_k=1,
            tool_vote_max_samples=2,
            tool_vote_max_model_calls=2,
        ),
        trace=trace,
    )
    result = agent.run("use dummy tool")
    assert "done" in result.answer
    assert any(event["type"] == "tool_result" for event in trace.events)
