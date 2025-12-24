"""Eval runner and trace replay utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    return registry


def build_model(settings: Settings, use_mock: bool):
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


def run_tasks(
    input_path: Path,
    output_path: Path,
    settings: Settings,
    verify: bool,
    self_consistency: int,
    max_model_calls: int | None,
    summary_lines: int | None,
    use_mock: bool = False,
) -> None:
    if summary_lines is not None:
        settings.summary_lines = summary_lines
    if max_model_calls is not None:
        settings.max_model_calls = max_model_calls
    model = build_model(settings, use_mock=use_mock)
    registry = build_registry(settings, model)
    policy = SafetyPolicy(max_model_calls=settings.max_model_calls)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            if not line.strip():
                continue
            payload = json.loads(line)
            task_id = payload.get("id") or uuid4().hex
            query = payload.get("query", "")
            trace = TraceRecorder(trace_id=str(task_id), workspace_dir=settings.workspace_dir)
            memory = MemoryStore(
                max_tool_output_chars=settings.max_tool_output_chars,
                keep_raw_tool_output=settings.keep_raw_tool_output,
                summary_lines=settings.summary_lines,
            )
            agent = Agent(
                model=model,
                registry=registry,
                policy=policy,
                mode=settings.agent_mode,
                verify=verify,
                self_consistency=self_consistency,
                max_model_calls=settings.max_model_calls,
                memory=memory,
                trace=trace,
            )
            result = agent.run(query)
            output = {
                "id": task_id,
                "answer": result.answer,
                "tool_calls": result.tools_used,
                "tools_created": result.tools_created,
                "trace_path": result.trace_path,
            }
            outfile.write(json.dumps(output) + "\n")


def replay_trace(trace_path: Path, registry: ToolRegistry) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    tool_events = [event for event in payload.get("events", []) if event.get("type") == "tool_call"]
    results = []
    for event in tool_events:
        tool_name = event["payload"]["tool_name"]
        arguments = event["payload"]["arguments"]
        tool = registry.get(tool_name)
        if not tool:
            results.append({"tool_name": tool_name, "error": "unknown tool"})
            continue
        result = tool.run(arguments)
        results.append({"tool_name": tool_name, "output": result.output})
    return {"trace_id": payload.get("trace_id"), "results": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgentForge eval runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run eval tasks")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--verify", action="store_true")
    run_parser.add_argument("--self-consistency", type=int, default=1)
    run_parser.add_argument("--max-model-calls", type=int)
    run_parser.add_argument("--summary-lines", type=int)
    run_parser.add_argument("--mock", action="store_true")

    replay_parser = subparsers.add_parser("replay", help="Replay a trace")
    replay_parser.add_argument("--trace", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()

    if args.command == "run":
        run_tasks(
            input_path=Path(args.input),
            output_path=Path(args.output),
            settings=settings,
            verify=args.verify,
            self_consistency=args.self_consistency,
            max_model_calls=args.max_model_calls,
            summary_lines=args.summary_lines,
            use_mock=args.mock,
        )
        return
    if args.command == "replay":
        registry = build_registry(settings, MockChatModel())
        result = replay_trace(Path(args.trace), registry)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
