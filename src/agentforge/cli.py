"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agentforge.agent import Agent
from agentforge.config import Settings
from agentforge.memory import MemoryStore
from agentforge.models.mock import MockChatModel
from agentforge.models.openai_compat import OpenAICompatChatModel
from agentforge.safety.policy import SafetyPolicy
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentForge CLI")
    parser.add_argument("query", type=str, help="Prompt to run")
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--mode", choices=["direct", "deep"], dest="mode")
    parser.add_argument("--allow-tool-creation", action="store_true", dest="allow_tool_creation")
    parser.add_argument("--workspace", dest="workspace")
    parser.add_argument("--verify", action="store_true", dest="verify")
    parser.add_argument("--self-consistency", type=int, dest="self_consistency", default=1)
    parser.add_argument("--max-model-calls", type=int, dest="max_model_calls")
    parser.add_argument("--summary-lines", type=int, dest="summary_lines")
    parser.add_argument("--strict-json", action="store_true", dest="strict_json")
    parser.add_argument("--code-check", action="store_true", dest="code_check")
    parser.add_argument("--code-check-max-iters", type=int, dest="code_check_max_iters")
    parser.add_argument("--max-message-chars", type=int, dest="max_message_chars")
    parser.add_argument("--max-single-message-chars", type=int, dest="max_single_message_chars")
    parser.add_argument("--max-turns", type=int, dest="max_turns")
    parser.add_argument("--profile", dest="profile", choices=["agent", "code", "math", "qa"])
    parser.add_argument("--branch-candidates", type=int, dest="branch_candidates")
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
    if args.summary_lines:
        data["summary_lines"] = args.summary_lines
    if args.max_model_calls:
        data["max_model_calls"] = args.max_model_calls
    if args.strict_json:
        data["strict_json_mode"] = True
    if args.code_check:
        data["code_check"] = True
    if args.code_check_max_iters:
        data["code_check_max_iters"] = args.code_check_max_iters
    if args.max_message_chars:
        data["max_message_chars"] = args.max_message_chars
    if args.max_single_message_chars:
        data["max_single_message_chars"] = args.max_single_message_chars
    if args.max_turns:
        data["max_turns"] = args.max_turns
    if args.branch_candidates is not None:
        data["branch_candidates"] = args.branch_candidates
    return Settings(**data)


def main() -> None:
    args = parse_args()
    settings = apply_overrides(Settings(), args)
    model = build_model(settings)
    registry = build_registry(settings, model)
    memory = MemoryStore(
        max_tool_output_chars=settings.max_tool_output_chars,
        keep_raw_tool_output=settings.keep_raw_tool_output,
        summary_lines=settings.summary_lines,
    )
    policy = SafetyPolicy(max_model_calls=settings.max_model_calls)
    verify_override = "--verify" in sys.argv
    strict_json_override = "--strict-json" in sys.argv or settings.strict_json_mode
    code_check_override = "--code-check" in sys.argv or settings.code_check
    code_check_iters_override = "--code-check-max-iters" in sys.argv
    agent = Agent(
        model=model,
        registry=registry,
        policy=policy,
        mode=settings.agent_mode,
        verify=True if verify_override else None,
        self_consistency=args.self_consistency,
        max_model_calls=settings.max_model_calls,
        memory=memory,
        strict_json_mode=True if strict_json_override else None,
        max_message_chars=settings.max_message_chars,
        max_message_tokens_approx=settings.max_message_tokens_approx,
        token_char_ratio=settings.token_char_ratio,
        max_single_message_chars=settings.max_single_message_chars,
        max_turns=settings.max_turns,
        trim_strategy=settings.trim_strategy,
        code_check=True if code_check_override else None,
        code_check_max_iters=(
            settings.code_check_max_iters
            if code_check_iters_override or settings.code_check_max_iters != 2
            else None
        ),
        profile=args.profile,
        branch_candidates=settings.branch_candidates,
    )
    result = agent.run(args.query)
    print("Tools used:", ", ".join(result.tools_used) or "none")
    print("Tools created:", ", ".join(result.tools_created) or "none")
    print("Verify enabled:", "yes" if args.verify else "no")
    print("Self-consistency:", args.self_consistency)
    print("Answer:\n", result.answer)


if __name__ == "__main__":
    main()
