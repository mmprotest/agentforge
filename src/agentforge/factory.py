"""Shared construction helpers for models, tools, and agents."""

from __future__ import annotations

import json
from typing import Any

from agentforge.agent import Agent
from agentforge.config import Settings
from agentforge.memory import MemoryStore
from agentforge.models.base import BaseChatModel
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


def build_model(settings: Settings, use_mock: bool = False) -> BaseChatModel:
    if use_mock or not settings.openai_api_key:
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


def build_registry(
    settings: Settings, model: BaseChatModel, include_tool_maker: bool | None = None
) -> ToolRegistry:
    registry = ToolRegistry()
    allowed_imports = {
        item.strip().lower()
        for item in (settings.sandbox_allowed_imports or "").split(",")
        if item.strip()
    }
    registry.register(HttpFetchTool())
    registry.register(FileSystemTool(settings.workspace_dir))
    registry.register(PythonSandboxTool(settings.workspace_dir, allowed_imports=allowed_imports))
    registry.register(DeepThinkTool())
    registry.register(CalculatorTool())
    registry.register(RegexExtractTool())
    registry.register(UnitConvertTool())
    registry.register(CodeRunMultiTool(settings.workspace_dir))
    registry.register(JsonRepairTool())
    allow_tool_maker = settings.allow_tool_creation if include_tool_maker is None else include_tool_maker
    if allow_tool_maker:
        maker = ToolMaker(model, settings.workspace_dir)
        registry.register(ToolMakerTool(maker, registry))
    return registry


def build_memory(settings: Settings) -> MemoryStore:
    return MemoryStore(
        max_tool_output_chars=settings.max_tool_output_chars,
        keep_raw_tool_output=settings.keep_raw_tool_output,
        summary_lines=settings.summary_lines,
    )


def build_policy(settings: Settings) -> SafetyPolicy:
    return SafetyPolicy(
        max_steps=settings.max_steps,
        max_tool_calls=settings.max_tool_calls,
        max_model_calls=settings.max_model_calls,
        tool_vote_enabled=settings.tool_vote_enabled,
        tool_vote_k=settings.tool_vote_k,
        tool_vote_max_samples=settings.tool_vote_max_samples,
        tool_vote_max_model_calls=settings.tool_vote_max_model_calls,
    )


def build_agent(
    settings: Settings,
    model: BaseChatModel,
    registry: ToolRegistry,
    *,
    verify: bool = False,
    self_consistency: int = 1,
    trace: TraceRecorder | None = None,
    memory: MemoryStore | None = None,
    overrides: dict[str, Any] | None = None,
) -> Agent:
    policy = build_policy(settings)
    return Agent(
        model=model,
        registry=registry,
        policy=policy,
        mode=settings.agent_mode,
        verify=verify,
        self_consistency=self_consistency,
        max_model_calls=settings.max_model_calls,
        max_steps=settings.max_steps,
        max_tool_calls=settings.max_tool_calls,
        memory=memory or build_memory(settings),
        trace=trace,
        strict_json_mode=settings.strict_json_mode,
        max_message_chars=settings.max_message_chars,
        max_turns=settings.max_turns,
        trim_strategy=settings.trim_strategy,
        code_check=settings.code_check,
        code_check_max_iters=settings.code_check_max_iters,
        **(overrides or {}),
    )
