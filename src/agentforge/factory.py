"""Factory helpers for shared construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentforge.agent import Agent
from agentforge.config import Settings
from agentforge.controller import Controller
from agentforge.models.base import BaseChatModel
from agentforge.models.mock import MockChatModel
from agentforge.models.openai_compat import OpenAICompatChatModel
from agentforge.state import AgentState
from agentforge.verifier import ConstraintVerifier, FormatVerifier


@dataclass
class Runtime:
    model: BaseChatModel
    registry: dict[str, Any]
    controller: Controller
    state: AgentState
    agent: Agent


def build_model(settings: Settings) -> BaseChatModel:
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


def build_runtime(
    settings: Settings,
    task: str,
    constraints: list[str] | None = None,
    artifacts: dict[str, Any] | None = None,
    seed: int = 0,
) -> Runtime:
    model = build_model(settings)
    registry: dict[str, Any] = {}
    verifiers = [FormatVerifier(), ConstraintVerifier()]
    controller = Controller(
        model=model,
        verifiers=verifiers,
        proposal_count=settings.proposal_count,
        max_backtracks=settings.max_backtracks,
        max_attempts=settings.max_attempts,
        seed=seed,
    )
    state = AgentState(
        task=task,
        constraints=constraints or [],
        artifacts=artifacts or {},
        history=[],
        attempts=0,
        branch_id="root",
        current_plan=None,
        verifier_results=[],
    )
    agent = Agent(controller=controller)
    return Runtime(
        model=model,
        registry=registry,
        controller=controller,
        state=state,
        agent=agent,
    )
