"""Smoke test helpers."""

from __future__ import annotations

import json
from pathlib import Path

from agentforge.config import Settings
from agentforge.factory import build_agent, build_model, build_registry
from agentforge.runtime.runtime import Runtime
from agentforge.workflows.engine import WorkflowEngine, load_workflow_spec


def run_smoke_test(settings: Settings, runtime: Runtime) -> None:
    model = build_model(settings, use_mock=True)
    registry = build_registry(settings, model)
    agent = build_agent(settings, model, registry, runtime=runtime)
    agent.run("2+2")
    calculator = registry.get("calculator")
    if calculator:
        calculator.run({"expression": "2+2"})
    python_tool = registry.get("python_sandbox")
    if python_tool:
        python_tool.run({"code": "import math\nmath.sqrt(4)", "timeout_seconds": 2})
    spec = {
        "name": "smoke",
        "version": "0.1.0",
        "description": "smoke test",
        "inputs_schema": {"type": "object"},
        "outputs_schema": {"type": "object"},
        "steps": [
            {
                "id": "calc",
                "kind": "tool",
                "tool_name": "calculator",
                "tool_args_template": {"expression": "2+2"},
                "outputs_key": "output",
            }
        ],
    }
    tmp_path = runtime.workspace.path / "smoke_workflow.json"
    tmp_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    engine = WorkflowEngine(model, registry, runtime=runtime)
    engine.run(load_workflow_spec(tmp_path), {})
    tmp_path.unlink()
