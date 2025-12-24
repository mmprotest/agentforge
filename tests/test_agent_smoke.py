from agentforge.agent import Agent
import json

from agentforge.models.base import ModelResponse
from agentforge.models.mock import MockChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.builtins.deep_think import DeepThinkTool
from agentforge.tools.builtins.filesystem import FileSystemTool
from agentforge.tools.builtins.http_fetch import HttpFetchTool
from agentforge.tools.builtins.python_sandbox import PythonSandboxTool
from agentforge.tools.registry import ToolRegistry


def test_agent_smoke(tmp_path):
    registry = ToolRegistry()
    registry.register(HttpFetchTool())
    registry.register(FileSystemTool(str(tmp_path)))
    registry.register(PythonSandboxTool(str(tmp_path)))
    registry.register(DeepThinkTool())
    model = MockChatModel()
    agent = Agent(model=model, registry=registry, policy=SafetyPolicy(), mode="direct")
    result = agent.run("hello")
    assert "Mock response" in result.answer


def test_deep_mode_does_not_add_internal_plan_message(tmp_path):
    registry = ToolRegistry()
    registry.register(DeepThinkTool())
    model = MockChatModel(
        scripted=[
            ModelResponse(
                final_text=json.dumps(
                    {
                        "type": "final",
                        "answer": "ok",
                        "confidence": 0.1,
                        "checks": [],
                    }
                )
            )
        ]
    )
    agent = Agent(
        model=model,
        registry=registry,
        policy=SafetyPolicy(max_model_calls=1),
        mode="deep",
    )
    result = agent.run("plan something")
    assert result.answer.startswith("ok")
    assert agent.memory.state.get("plan") is not None
    assert all(
        "Internal plan" not in str(message.get("content", ""))
        for message in agent._messages
    )
