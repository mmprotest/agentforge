"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agentforge.config import Settings
from agentforge.factory import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentForge CLI")
    parser.add_argument("query", type=str, help="Prompt to run")
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--proposal-count", type=int, dest="proposal_count")
    parser.add_argument("--max-backtracks", type=int, dest="max_backtracks")
    parser.add_argument("--max-attempts", type=int, dest="max_attempts")
    parser.add_argument("--constraints", type=str, dest="constraints")
    parser.add_argument("--require-json", action="store_true", dest="require_json")
    parser.add_argument("--output-schema", type=str, dest="output_schema")
    return parser.parse_args()


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    data: dict[str, Any] = settings.model_dump()
    if args.base_url:
        data["openai_base_url"] = args.base_url
    if args.api_key:
        data["openai_api_key"] = args.api_key
    if args.model:
        data["openai_model"] = args.model
    if args.proposal_count is not None:
        data["proposal_count"] = args.proposal_count
    if args.max_backtracks is not None:
        data["max_backtracks"] = args.max_backtracks
    if args.max_attempts is not None:
        data["max_attempts"] = args.max_attempts
    return Settings(**data)


def main() -> None:
    args = parse_args()
    settings = apply_overrides(Settings(), args)
    constraints = []
    if args.constraints:
        constraints = [item.strip() for item in args.constraints.split(";") if item.strip()]
    artifacts: dict[str, Any] = {}
    if args.require_json:
        artifacts["require_json"] = True
    if args.output_schema:
        artifacts["output_schema"] = json.loads(args.output_schema)
    runtime = build_runtime(settings, task=args.query, constraints=constraints, artifacts=artifacts)
    result = runtime.agent.run(runtime.state)
    print(result.answer)


if __name__ == "__main__":
    main()
