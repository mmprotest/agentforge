"""Eval runner and trace replay utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentforge.config import Settings
from agentforge.factory import build_agent, build_model, build_registry
from agentforge.trace import TraceRecorder
from agentforge.models.mock import MockChatModel
from agentforge.tools.registry import ToolRegistry


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
    registry = build_registry(settings, model, include_tool_maker=False)

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
            agent = build_agent(
                settings,
                model,
                registry,
                verify=verify,
                self_consistency=self_consistency,
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
        registry = build_registry(settings, MockChatModel(), include_tool_maker=False)
        result = replay_trace(Path(args.trace), registry)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
