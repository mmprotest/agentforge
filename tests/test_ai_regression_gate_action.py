from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentforge.config import Settings
from agentforge.evals import action as gate_action


def _load_entrypoint_module():
    entrypoint_path = Path(__file__).resolve().parents[1] / ".github" / "actions" / "ai-regression-gate" / "entrypoint.py"
    spec = spec_from_file_location("ai_regression_gate_entrypoint", entrypoint_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_eval_command_uses_global_workspace_flag() -> None:
    entrypoint = _load_entrypoint_module()
    report_path = Path("agentforge_eval_report.json")
    command = entrypoint._build_eval_command(
        workspace_id="demo",
        pack_name="sample",
        report_path=report_path,
        agentforge_args=["--foo", "bar"],
    )
    assert command[:3] == ["agentforge", "--workspace", "demo"]
    assert command[3:8] == ["eval", "run", "--pack", "sample", "--report"]
    assert command[8] == str(report_path)
    assert command[-2:] == ["--foo", "bar"]


def test_action_requires_baseline_in_pr_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INPUT_WORKSPACE", "default")
    monkeypatch.setenv("INPUT_EVAL_PACK", "sample")
    monkeypatch.setenv("INPUT_BASELINE_STRATEGY", "artifact")
    monkeypatch.setenv("INPUT_BASELINE_ARTIFACT_NAME", "ai-baseline-report")
    monkeypatch.setenv("INPUT_BASELINE_ARTIFACT_PATH", "baseline_report.json")
    monkeypatch.setenv("INPUT_MIN_SCORE", "0.0")
    monkeypatch.setenv("INPUT_ALLOW_REGRESSION", "false")
    monkeypatch.setenv("INPUT_FAIL_ON_MISSING_BASELINE", "true")
    monkeypatch.setenv("INPUT_REPORT_OUT", str(tmp_path / "candidate.json"))
    monkeypatch.setenv("INPUT_MODE", "auto")
    monkeypatch.setenv("INPUT_AGENTFORGE_ARGS", "")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

    workspace_root = tmp_path / "workspaces" / "default" / "evals" / "sample"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "pack.jsonl").write_text("", encoding="utf-8")

    settings = Settings(
        openai_api_key="test-key",
        openai_base_url="http://example.com",
        openai_model="gpt-test",
    )
    runtime = SimpleNamespace(workspace=SimpleNamespace(path=tmp_path / "workspaces" / "default"))

    monkeypatch.setattr(gate_action, "_build_runtime", lambda *_: (settings, runtime))
    monkeypatch.setattr(
        gate_action,
        "_prepare_baseline",
        lambda *_: (_ for _ in ()).throw(
            gate_action.BaselineError(gate_action._baseline_missing_message())
        ),
    )

    with pytest.raises(SystemExit) as exc:
        gate_action.main()
    assert str(exc.value) == gate_action._baseline_missing_message()
