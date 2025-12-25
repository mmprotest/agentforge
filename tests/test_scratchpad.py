from pathlib import Path

from agentforge.agent import Agent
from agentforge.models.base import ModelResponse
from agentforge.models.mock import MockChatModel
from agentforge.tools.registry import ToolRegistry
from agentforge.trace import TraceRecorder


def test_scratchpad_is_never_in_stdout(tmp_path: Path):
    model = MockChatModel(
        scripted=[
            ModelResponse(
                final_text=(
                    '{"type":"final","answer":"ok","scratchpad":"secret",'
                    '"confidence":0.2,"checks":[]}'
                )
            ),
            ModelResponse(
                final_text=(
                    '{"type":"final","answer":"ok","scratchpad":"secret",'
                    '"confidence":0.2,"checks":[]}'
                )
            ),
            ModelResponse(
                final_text=(
                    '{"type":"final","answer":"ok","scratchpad":"secret",'
                    '"confidence":0.2,"checks":[]}'
                )
            ),
        ]
    )
    agent = Agent(model=model, registry=ToolRegistry(), eval_mode=True)
    result = agent.run("hello")
    assert result.answer == "ok"
    assert "secret" not in result.answer


def test_trace_contains_scratchpad_when_present(tmp_path: Path):
    model = MockChatModel(
        scripted=[
            ModelResponse(
                final_text=(
                    '{"type":"final","answer":"ok","scratchpad":"secret",'
                    '"confidence":0.2,"checks":[]}'
                )
            ),
            ModelResponse(
                final_text=(
                    '{"type":"final","answer":"ok","scratchpad":"secret",'
                    '"confidence":0.2,"checks":[]}'
                )
            ),
            ModelResponse(
                final_text=(
                    '{"type":"final","answer":"ok","scratchpad":"secret",'
                    '"confidence":0.2,"checks":[]}'
                )
            ),
        ]
    )
    trace = TraceRecorder(trace_id="scratch", workspace_dir=str(tmp_path))
    agent = Agent(model=model, registry=ToolRegistry(), trace=trace)
    result = agent.run("hello")
    assert result.trace_path
    payload = Path(result.trace_path).read_text(encoding="utf-8")
    assert "secret" in payload
