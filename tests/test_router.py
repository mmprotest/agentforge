import json

from agentforge.agent import Agent
from agentforge.models.base import BaseChatModel, ModelResponse
from agentforge.models.mock import MockChatModel
from agentforge.routing import suggest_tool
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.base import Tool, ToolResult
from agentforge.tools.registry import ToolRegistry


def test_router_detects_url():
    suggestion = suggest_tool("User query: Fetch https://example.com")
    assert suggestion is not None
    assert suggestion.tool_name == "http_fetch"


def test_router_detects_arithmetic():
    suggestion = suggest_tool("User query: What is 2+2?")
    assert suggestion is not None
    assert suggestion.tool_name == "calculator"


def test_router_prompt_includes_last_tool_summary():
    agent = Agent(model=MockChatModel(), registry=ToolRegistry())
    agent.memory.state["facts"].append("use cached data")
    prompt = agent._build_router_prompt("What now?", "tool summary here", None)
    assert "tool summary here" in prompt
    assert "use cached data" in prompt


def test_router_direct_http_fetch_without_model_call(tmp_path):
    class CountingModel(BaseChatModel):
        def __init__(self) -> None:
            self.calls = 0

        def chat(
            self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None
        ) -> ModelResponse:
            self.calls += 1
            return ModelResponse(
                final_text=json.dumps(
                    {"type": "final", "answer": "done", "confidence": 0.1, "checks": []}
                )
            )

    class DummyHttpFetchTool(Tool):
        name = "http_fetch"
        description = "dummy"

        def run(self, data: object) -> ToolResult:
            return ToolResult(output={"status_code": 200, "body": "ok"})

    registry = ToolRegistry()
    registry.register(DummyHttpFetchTool())
    model = CountingModel()
    agent = Agent(
        model=model,
        registry=registry,
        policy=SafetyPolicy(max_model_calls=0),
    )
    result = agent.run("Fetch https://example.com")
    assert "http_fetch" in result.tools_used
    assert model.calls == 0
