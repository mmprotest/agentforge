"""Workflow specification definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowStepAcceptance:
    require_schema: bool = False
    require_citations: bool = False
    regex: str | None = None
    max_tokens: int | None = None


@dataclass
class WorkflowStepRetry:
    max_attempts: int = 1
    on_fail: str = "stop"


@dataclass
class WorkflowStep:
    id: str
    kind: str
    tool_name: str | None = None
    tool_args_template: dict[str, Any] | None = None
    llm_prompt_template: str | None = None
    acceptance: WorkflowStepAcceptance = field(default_factory=WorkflowStepAcceptance)
    retry: WorkflowStepRetry = field(default_factory=WorkflowStepRetry)
    outputs_key: str = "output"


@dataclass
class WorkflowSpec:
    name: str
    version: str
    description: str
    inputs_schema: dict[str, Any]
    outputs_schema: dict[str, Any]
    steps: list[WorkflowStep]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkflowSpec":
        steps_payload = payload.get("steps", [])
        steps: list[WorkflowStep] = []
        for item in steps_payload:
            acceptance_payload = item.get("acceptance") or {}
            retry_payload = item.get("retry") or {}
            steps.append(
                WorkflowStep(
                    id=item["id"],
                    kind=item["kind"],
                    tool_name=item.get("tool_name"),
                    tool_args_template=item.get("tool_args_template"),
                    llm_prompt_template=item.get("llm_prompt_template"),
                    acceptance=WorkflowStepAcceptance(
                        require_schema=acceptance_payload.get("require_schema", False),
                        require_citations=acceptance_payload.get("require_citations", False),
                        regex=acceptance_payload.get("regex"),
                        max_tokens=acceptance_payload.get("max_tokens"),
                    ),
                    retry=WorkflowStepRetry(
                        max_attempts=retry_payload.get("max_attempts", 1),
                        on_fail=retry_payload.get("on_fail", "stop"),
                    ),
                    outputs_key=item.get("outputs_key", "output"),
                )
            )
        return cls(
            name=payload["name"],
            version=payload["version"],
            description=payload.get("description", ""),
            inputs_schema=payload.get("inputs_schema", {}),
            outputs_schema=payload.get("outputs_schema", {}),
            steps=steps,
        )
