"""Minimal benchmark runner."""

from __future__ import annotations

import argparse
import json

from agentforge.config import Settings
from agentforge.factory import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentForge benchmark runner")
    parser.add_argument("--task-file", required=True, help="Path to JSON task file")
    parser.add_argument("--output", required=True, help="Path to write result JSON")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_task(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    payload = load_task(args.task_file)
    settings = Settings()
    artifacts = payload.get("artifacts", {})
    if payload.get("require_json") is not None:
        artifacts["require_json"] = payload["require_json"]
    if payload.get("output_schema") is not None:
        artifacts["output_schema"] = payload["output_schema"]
    runtime = build_runtime(
        settings,
        task=payload["task"],
        constraints=payload.get("constraints", []),
        artifacts=artifacts,
        seed=args.seed,
    )
    result = runtime.agent.run(runtime.state)
    summary = {
        "task": payload["task"],
        "constraints": payload.get("constraints", []),
        "answer": result.answer,
        "branch_id": result.state.branch_id,
        "attempts": result.state.attempts,
        "proposals": result.state.artifacts.get("proposals", []),
        "selected_branch": result.state.artifacts.get("selected_branch"),
        "selected_action": result.state.artifacts.get("selected_action"),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
