from agentforge.agent import Agent
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
