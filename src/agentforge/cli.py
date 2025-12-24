"""Command-line interface."""

from __future__ import annotations

import argparse
from typing import Any

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


def build_model(settings: Settings):
    if not settings.openai_api_key:
        return MockChatModel()
    return OpenAICompatChatModel(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentForge CLI")
    parser.add_argument("query", type=str, help="Prompt to run")
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--mode", choices=["direct", "deep"], dest="mode")
    parser.add_argument("--allow-tool-creation", action="store_true", dest="allow_tool_creation")
    parser.add_argument("--workspace", dest="workspace")
    return parser.parse_args()


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    data: dict[str, Any] = settings.model_dump()
    if args.base_url:
        data["openai_base_url"] = args.base_url
    if args.api_key:
        data["openai_api_key"] = args.api_key
    if args.model:
        data["openai_model"] = args.model
    if args.mode:
        data["agent_mode"] = args.mode
    if args.allow_tool_creation:
        data["allow_tool_creation"] = True
    if args.workspace:
        data["workspace_dir"] = args.workspace
    return Settings(**data)


def main() -> None:
    args = parse_args()
    settings = apply_overrides(Settings(), args)
    model = build_model(settings)
    registry = build_registry(settings, model)
    agent = Agent(model=model, registry=registry, policy=SafetyPolicy(), mode=settings.agent_mode)
    result = agent.run(args.query)
    print("Tools used:", ", ".join(result.tools_used) or "none")
    print("Tools created:", ", ".join(result.tools_created) or "none")
    print("Answer:\n", result.answer)


if __name__ == "__main__":
    main()
