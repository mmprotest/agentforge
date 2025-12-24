"""FastAPI service."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from agentforge.agent import Agent
from agentforge.config import Settings
from agentforge.models.mock import MockChatModel
from agentforge.models.openai_compat import OpenAICompatChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.builtins.deep_think import DeepThinkTool
from agentforge.tools.builtins.filesystem import FileSystemTool
from agentforge.tools.builtins.http_fetch import HttpFetchTool
from agentforge.tools.builtins.python_sandbox import PythonSandboxTool
from agentforge.tools.registry import ToolRegistry
from agentforge.tools.tool_maker import ToolMaker, ToolMakerTool

app = FastAPI()


class RunRequest(BaseModel):
    query: str
    mode: str | None = None
    allow_tool_creation: bool | None = None
    base_url: str | None = None
    model: str | None = None


class RunResponse(BaseModel):
    answer: str
    tool_calls: list[str]
    tools_created: list[str]


def build_model(settings: Settings):
    if not settings.openai_api_key:
        return MockChatModel()
    return OpenAICompatChatModel(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
    )


def build_registry(settings: Settings, model) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(HttpFetchTool())
    registry.register(FileSystemTool(settings.workspace_dir))
    registry.register(PythonSandboxTool(settings.workspace_dir))
    registry.register(DeepThinkTool())
    if settings.allow_tool_creation:
        maker = ToolMaker(model, settings.workspace_dir)
        registry.register(ToolMakerTool(maker, registry))
    return registry


@app.post("/run", response_model=RunResponse)
async def run_agent(request: RunRequest) -> RunResponse:
    settings = Settings()
    if request.base_url:
        settings.openai_base_url = request.base_url
    if request.model:
        settings.openai_model = request.model
    if request.mode:
        settings.agent_mode = request.mode
    if request.allow_tool_creation is not None:
        settings.allow_tool_creation = request.allow_tool_creation
    model = build_model(settings)
    registry = build_registry(settings, model)
    agent = Agent(model=model, registry=registry, policy=SafetyPolicy(), mode=settings.agent_mode)
    result = agent.run(request.query)
    return RunResponse(
        answer=result.answer, tool_calls=result.tools_used, tools_created=result.tools_created
    )
