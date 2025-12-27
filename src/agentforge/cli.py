"""Command-line interface."""

from __future__ import annotations

import argparse
from typing import Any

from agentforge.config import Settings
from agentforge.factory import build_agent, build_model, build_registry


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
    parser.add_argument("--max-steps", type=int, dest="max_steps")
    parser.add_argument("--max-tool-calls", type=int, dest="max_tool_calls")
    parser.add_argument("--max-model-calls", type=int, dest="max_model_calls")
    parser.add_argument("--summary-lines", type=int, dest="summary_lines")
    parser.add_argument("--strict-json", action="store_true", dest="strict_json")
    parser.add_argument("--code-check", action="store_true", dest="code_check")
    parser.add_argument("--code-check-max-iters", type=int, dest="code_check_max_iters")
    parser.add_argument("--max-message-chars", type=int, dest="max_message_chars")
    parser.add_argument("--max-turns", type=int, dest="max_turns")
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
    if args.max_steps:
        data["max_steps"] = args.max_steps
    if args.max_tool_calls:
        data["max_tool_calls"] = args.max_tool_calls
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
    if args.max_turns:
        data["max_turns"] = args.max_turns
    return Settings(**data)


def main() -> None:
    args = parse_args()
    settings = apply_overrides(Settings(), args)
    model = build_model(settings)
    registry = build_registry(settings, model)
    agent = build_agent(
        settings,
        model,
        registry,
        verify=args.verify,
        self_consistency=args.self_consistency,
    )
    result = agent.run(args.query)
    print("Tools used:", ", ".join(result.tools_used) or "none")
    print("Tools created:", ", ".join(result.tools_created) or "none")
    print("Verify enabled:", "yes" if args.verify else "no")
    print("Self-consistency:", args.self_consistency)
    print("Answer:\n", result.answer)


if __name__ == "__main__":
    main()
