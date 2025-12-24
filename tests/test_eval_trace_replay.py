from pathlib import Path

from agentforge.agent import Agent
from agentforge.config import Settings
from agentforge.eval.runner import build_registry, replay_trace
from agentforge.memory import MemoryStore
from agentforge.models.base import ModelResponse, ToolCall
from agentforge.models.mock import MockChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.trace import TraceRecorder


def test_trace_and_replay(tmp_path):
    settings = Settings(workspace_dir=str(tmp_path))
    model = MockChatModel(
        scripted=[
            ModelResponse(tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"})),
            ModelResponse(final_text='{"type":"final","answer":"4","confidence":1,"checks":["calc"]}'),
        ]
    )
    registry = build_registry(settings, model)
    trace = TraceRecorder(trace_id="test-trace", workspace_dir=str(tmp_path))
    memory = MemoryStore(summary_lines=5)
    agent = Agent(
        model=model,
        registry=registry,
        policy=SafetyPolicy(),
        memory=memory,
        trace=trace,
    )
    result = agent.run("2+2")
    assert result.trace_path
    trace_path = Path(result.trace_path)
    assert trace_path.exists()
    replay_result = replay_trace(trace_path, registry)
    assert replay_result["results"]
    assert replay_result["results"][0]["tool_name"] == "calculator"
