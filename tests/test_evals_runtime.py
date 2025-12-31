import json
from pathlib import Path

from agentforge.config import Settings
from agentforge.evals.runner import run_eval_pack
from agentforge.factory import build_agent, build_model, build_registry
from agentforge.runtime.runtime import Runtime
from agentforge.workflows.engine import WorkflowEngine


def test_eval_pack_runs_agent_and_workflow(tmp_path: Path) -> None:
    runtime = Runtime.from_workspace(tmp_path, "default")
    settings = Settings(workspace_dir=str(runtime.workspace.path))
    model = build_model(settings, use_mock=True)
    registry = build_registry(settings, model)
    agent = build_agent(settings, model, registry, runtime=runtime)
    engine = WorkflowEngine(model, registry, runtime=runtime)

    workflow_path = runtime.workspace.path / "simple_workflow.json"
    workflow_payload = {
        "name": "wf",
        "version": "1.0.0",
        "description": "simple",
        "inputs_schema": {"type": "object"},
        "outputs_schema": {"type": "string"},
        "steps": [
            {
                "id": "tmpl",
                "kind": "template",
                "llm_prompt_template": "hello",
                "outputs_key": "output",
            }
        ],
    }
    workflow_path.write_text(json.dumps(workflow_payload), encoding="utf-8")

    pack_dir = runtime.workspace.path / "evals" / "sample"
    pack_dir.mkdir(parents=True)
    pack_path = pack_dir / "pack.jsonl"
    pack_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "agent-1",
                        "input": "hello",
                        "expected_output": "Mock response to:",
                        "mode": "agent",
                        "scoring": "contains",
                    }
                ),
                json.dumps(
                    {
                        "id": "workflow-1",
                        "input": {},
                        "expected_output": "hello",
                        "mode": "workflow",
                        "workflow": str(workflow_path),
                        "scoring": "contains",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report = run_eval_pack(pack_path, agent, engine, report_path)
    assert report["overall_score"] == 1.0
