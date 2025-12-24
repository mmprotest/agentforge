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
        policy=SafetyPolicy(max_model_calls=5),
        trace=trace,
    )
    result = agent.run("use dummy tool")
    assert "done" in result.answer
    assert any(
        event["type"] == "tool_result" and '"ok": false' in event["payload"]["summary"]
        for event in trace.events
    )


def test_tool_error_recovery_includes_retry_instruction():
    class CaptureModel(MockChatModel):
        def __init__(self, scripted: list[ModelResponse]) -> None:
            super().__init__(scripted=scripted)
            self.seen_messages: list[list[dict[str, object]]] = []

        def chat(
            self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None
        ) -> ModelResponse:
            self.seen_messages.append(messages)
            return super().chat(messages, tools)

    scripted = [
        ModelResponse(tool_call=ToolCall(name="dummy", arguments={})),
        ModelResponse(
            final_text='{"type":"final","answer":"ok","confidence":0.2,"checks":[]}'
        ),
    ]
    model = CaptureModel(scripted=scripted)
    registry = ToolRegistry()
    registry.register(DummyTool())
    agent = Agent(model=model, registry=registry, policy=SafetyPolicy(max_model_calls=5))
    result = agent.run("use dummy tool")
    assert "ok" in result.answer
    assert any(
        "Return ONLY a tool call JSON object with corrected arguments."
        in str(message.get("content"))
        and "Required fields: value" in str(message.get("content"))
        and '"name": "dummy"' in str(message.get("content"))
        for call in model.seen_messages[1:]
        for message in call
    )
