"""FastAPI service."""

from __future__ import annotations

import json

from fastapi import FastAPI
from pydantic import BaseModel

from agentforge.config import Settings
from agentforge.factory import build_runtime

app = FastAPI()


class RunRequest(BaseModel):
    query: str
    base_url: str | None = None
    model: str | None = None
    proposal_count: int | None = None
    max_backtracks: int | None = None
    max_attempts: int | None = None
    constraints: list[str] | None = None
    require_json: bool | None = None
    output_schema: dict | None = None


class RunResponse(BaseModel):
    answer: str
    branch_id: str
    attempts: int


@app.post("/run", response_model=RunResponse)
async def run_agent(request: RunRequest) -> RunResponse:
    settings = Settings()
    if request.base_url:
        settings.openai_base_url = request.base_url
    if request.model:
        settings.openai_model = request.model
    if request.proposal_count is not None:
        settings.proposal_count = request.proposal_count
    if request.max_backtracks is not None:
        settings.max_backtracks = request.max_backtracks
    if request.max_attempts is not None:
        settings.max_attempts = request.max_attempts
    artifacts = {}
    if request.require_json is not None:
        artifacts["require_json"] = request.require_json
    if request.output_schema is not None:
        artifacts["output_schema"] = json.loads(json.dumps(request.output_schema))
    runtime = build_runtime(
        settings,
        task=request.query,
        constraints=request.constraints or [],
        artifacts=artifacts,
    )
    result = runtime.agent.run(runtime.state)
    return RunResponse(
        answer=result.answer,
        branch_id=result.state.branch_id,
        attempts=result.state.attempts,
    )
