"""FastAPI service."""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel

from agentforge.config import Settings
from agentforge.factory import build_agent, build_model, build_registry
from agentforge.trace import TraceRecorder

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
    trace = TraceRecorder(trace_id=f"api-{uuid4().hex[:8]}", workspace_dir=settings.workspace_dir)
    agent = build_agent(
        settings,
        model,
        registry,
        verify=bool(request.verify),
        self_consistency=request.self_consistency or 1,
        trace=trace,
    )
    result = agent.run(request.query)
    return RunResponse(
        answer=result.answer,
        tool_calls=result.tools_used,
        tools_created=result.tools_created,
        trace_path=result.trace_path,
    )
