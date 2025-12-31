import json
from pathlib import Path

import pytest

from agentforge.config import Settings
from agentforge.factory import build_model, build_registry
from agentforge.runtime.runtime import Runtime
from agentforge.workflows.engine import WorkflowEngine, WorkflowError
from agentforge.workflows.spec import WorkflowSpec


def test_workflow_tool_step_and_metrics(tmp_path: Path) -> None:
    runtime = Runtime.from_workspace(tmp_path, "default")
    settings = Settings(workspace_dir=str(runtime.workspace.path))
    model = build_model(settings, use_mock=True)
    registry = build_registry(settings, model)
    engine = WorkflowEngine(model, registry, runtime=runtime)
    spec_payload = {
        "name": "sum",
        "version": "1.0.0",
        "description": "add numbers",
        "inputs_schema": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
        "outputs_schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        "steps": [
            {
                "id": "calc",
                "kind": "tool",
                "tool_name": "calculator",
                "tool_args_template": {"expression": "{input.x}+{input.y}"},
                "outputs_key": "output",
            }
        ],
    }
    spec = WorkflowSpec.from_dict(spec_payload)
    result = engine.run(spec, {"x": 2, "y": 3}, runtime.new_run_context())
    assert result.outputs["value"] == "5"
    metrics_files = list((runtime.workspace.path / "metrics").glob("*.jsonl"))
    assert metrics_files
    last = json.loads(metrics_files[0].read_text(encoding="utf-8").splitlines()[-1])
    assert "run_id" in last and "duration_ms" in last


def test_workflow_acceptance_retry_failure(tmp_path: Path) -> None:
    runtime = Runtime.from_workspace(tmp_path, "default")
    settings = Settings(workspace_dir=str(runtime.workspace.path))
    model = build_model(settings, use_mock=True)
    registry = build_registry(settings, model)
    engine = WorkflowEngine(model, registry, runtime=runtime)
    spec_payload = {
        "name": "fail",
        "version": "1.0.0",
        "description": "fail acceptance",
        "inputs_schema": {"type": "object"},
        "outputs_schema": {"type": "string"},
        "steps": [
            {
                "id": "tmpl",
                "kind": "template",
                "llm_prompt_template": "no",
                "acceptance": {"regex": "yes"},
                "retry": {"max_attempts": 2, "on_fail": "retry"},
                "outputs_key": "output",
            }
        ],
    }
    spec = WorkflowSpec.from_dict(spec_payload)
    with pytest.raises(WorkflowError):
        engine.run(spec, {})
