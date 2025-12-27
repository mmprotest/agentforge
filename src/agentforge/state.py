"""Typed agent state for controller orchestration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    task: str
    constraints: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    history: list[str] = Field(default_factory=list)
    attempts: int = 0
    branch_id: str = "root"
    current_plan: str | None = None
    verifier_results: list["VerifierResult"] = Field(default_factory=list)


class Proposal(BaseModel):
    action: str
    rationale: str


class VerifierResult(BaseModel):
    pass_fail: bool
    reason: str
    verifier: str


AgentState.model_rebuild()
