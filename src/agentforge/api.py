"""FastAPI service."""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel

from agentforge.agent import Agent
from agentforge.config import Settings
from agentforge.memory import MemoryStore
from agentforge.models.mock import MockChatModel
from agentforge.models.openai_compat import OpenAICompatChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.trace import TraceRecorder
from agentforge.tools.builtins.calculator import CalculatorTool
from agentforge.tools.builtins.code_run_multi import CodeRunMultiTool
from agentforge.tools.builtins.deep_think import DeepThinkTool
from agentforge.tools.builtins.filesystem import FileSystemTool
from agentforge.tools.builtins.http_fetch import HttpFetchTool
from agentforge.tools.builtins.json_repair import JsonRepairTool
from agentforge.tools.builtins.python_sandbox import PythonSandboxTool
from agentforge.tools.builtins.regex_extract import RegexExtractTool
from agentforge.tools.builtins.unit_convert import UnitConvertTool
from agentforge.tools.registry import ToolRegistry
from agentforge.tools.tool_maker import ToolMaker, ToolMakerTool

app = FastAPI()


class RunRequest(BaseModel):
    query: str
    mode: str | None = None
    allow_tool_creation: bool | None = None
    base_url: str | None = None
    model: str | None = None
    verify: bool | None = None
    self_consistency: int | None = None
    max_steps: int | None = None
    max_tool_calls: int | None = None
    max_model_calls: int | None = None
    summary_lines: int | None = None
    strict_json: bool | None = None
    code_check: bool | None = None
    code_check_max_iters: int | None = None
    max_message_chars: int | None = None
    max_turns: int | None = None


class RunResponse(BaseModel):
    answer: str
    tool_calls: list[str]
    tools_created: list[str]
    trace_path: str | None = None


def build_model(settings: Settings):
    if not settings.openai_api_key:
        return MockChatModel()
    extra_headers = None
    if settings.openai_extra_headers:
        extra_headers = json.loads(settings.openai_extra_headers)
    return OpenAICompatChatModel(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
        extra_headers=extra_headers,
        disable_tool_choice=settings.openai_disable_tool_choice,
        force_chatcompletions_path=settings.openai_force_chatcompletions_path,
    )


def build_registry(settings: Settings, model) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(HttpFetchTool())
    registry.register(FileSystemTool(settings.workspace_dir))
    registry.register(PythonSandboxTool(settings.workspace_dir))
    registry.register(DeepThinkTool())
    registry.register(CalculatorTool())
    registry.register(RegexExtractTool())
    registry.register(UnitConvertTool())
    registry.register(CodeRunMultiTool(settings.workspace_dir))
    registry.register(JsonRepairTool())
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
    if request.summary_lines is not None:
        settings.summary_lines = request.summary_lines
    if request.max_steps is not None:
        settings.max_steps = request.max_steps
    if request.max_tool_calls is not None:
        settings.max_tool_calls = request.max_tool_calls
    if request.max_model_calls is not None:
        settings.max_model_calls = request.max_model_calls
    if request.strict_json is not None:
        settings.strict_json_mode = request.strict_json
    if request.code_check is not None:
        settings.code_check = request.code_check
    if request.code_check_max_iters is not None:
        settings.code_check_max_iters = request.code_check_max_iters
    if request.max_message_chars is not None:
        settings.max_message_chars = request.max_message_chars
    if request.max_turns is not None:
        settings.max_turns = request.max_turns
    model = build_model(settings)
    registry = build_registry(settings, model)
    memory = MemoryStore(
        max_tool_output_chars=settings.max_tool_output_chars,
        keep_raw_tool_output=settings.keep_raw_tool_output,
        summary_lines=settings.summary_lines,
    )
    policy = SafetyPolicy(
        max_steps=settings.max_steps,
        max_tool_calls=settings.max_tool_calls,
        max_model_calls=settings.max_model_calls,
    )
    trace = TraceRecorder(trace_id=f"api-{uuid4().hex[:8]}", workspace_dir=settings.workspace_dir)
    agent = Agent(
        model=model,
        registry=registry,
        policy=policy,
        mode=settings.agent_mode,
        verify=bool(request.verify),
        self_consistency=request.self_consistency or 1,
        max_model_calls=settings.max_model_calls,
        max_steps=settings.max_steps,
        max_tool_calls=settings.max_tool_calls,
        memory=memory,
        trace=trace,
        strict_json_mode=settings.strict_json_mode,
        max_message_chars=settings.max_message_chars,
        max_turns=settings.max_turns,
        trim_strategy=settings.trim_strategy,
        code_check=settings.code_check,
        code_check_max_iters=settings.code_check_max_iters,
    )
    result = agent.run(request.query)
    return RunResponse(
        answer=result.answer,
        tool_calls=result.tools_used,
        tools_created=result.tools_created,
        trace_path=result.trace_path,
    )
