import json

from agentforge.agent import Agent
from agentforge.models.base import BaseChatModel, ModelResponse
from agentforge.models.mock import MockChatModel
from agentforge.routing import suggest_tool, suggest_tools
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.base import Tool, ToolResult
from agentforge.tools.registry import ToolRegistry
from pydantic import BaseModel


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


def test_router_hints_are_ephemeral():
    class CaptureModel(BaseChatModel):
        def __init__(self, scripted: list[ModelResponse]) -> None:
            self.scripted = scripted
            self.seen_messages: list[list[dict[str, object]]] = []

        def chat(
            self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None
        ) -> ModelResponse:
            self.seen_messages.append(messages)
            return self.scripted.pop(0)

    class DummyInput(BaseModel):
        value: int

    class DummyTool(Tool):
        name = "dummy"
        description = "dummy tool"
        input_schema = DummyInput

        def run(self, data: BaseModel) -> ToolResult:
            payload = DummyInput.model_validate(data)
            return ToolResult(output={"ok": True, "value": payload.value})

    scripted = [
        ModelResponse(final_text='{"type":"final","answer":"step1","confidence":0.2,"checks":[]}'),
        ModelResponse(final_text='{"type":"final","answer":"step2","confidence":0.2,"checks":[]}'),
        ModelResponse(final_text='{"type":"final","answer":"done","confidence":0.2,"checks":[]}'),
    ]
    model = CaptureModel(scripted=scripted)
    registry = ToolRegistry()
    registry.register(DummyTool())
    agent = Agent(
        model=model,
        registry=registry,
        policy=SafetyPolicy(max_model_calls=5),
        profile="qa",
        verify=False,
    )
    result = agent.run("Use regex on foo bar")
    assert "done" in result.answer
    assert any(
        "Router hint" in str(message.get("content"))
        for call in model.seen_messages
        for message in call
        if message.get("role") == "user"
    )
    assert all(
        "Router hint" not in str(message.get("content"))
        for message in agent._messages
    )


def test_regex_extract_builder_prefers_slash_pattern_and_quoted_text():
    agent = Agent(model=MockChatModel(), registry=ToolRegistry())
    query = 'Extract /foo\\d+/ from "prefix foo123 suffix"'
    args = agent._direct_tool_args("regex_extract", query, None)
    assert args is not None
    assert args["pattern"] == "foo\\d+"
    assert args["text"] == "prefix foo123 suffix"


def test_router_does_not_trigger_conversion_on_generic_to():
    suggestions = suggest_tools("User query: I'd like to go to the store")
    assert all(suggestion.tool_name != "unit_convert" for suggestion in suggestions)


def test_router_triggers_conversion_on_convert_with_units():
    suggestion = suggest_tool("User query: convert 5 km to m")
    assert suggestion is not None
    assert suggestion.tool_name == "unit_convert"


def test_router_requires_path_or_file_intent_for_filesystem():
    suggestions = suggest_tools("User query: file this idea for later")
    assert all(suggestion.tool_name != "filesystem" for suggestion in suggestions)
    suggestions = suggest_tools("User query: read file /tmp/example.txt")
    assert any(suggestion.tool_name == "filesystem" for suggestion in suggestions)
