from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentforge import cli


def _write_report(path: Path, score: float) -> None:
    report = {"report_version": "0.1", "overall_score": score, "failures": []}
    path.write_text(json.dumps(report), encoding="utf-8")


def _prepare_pack(tmp_path: Path, pack_name: str) -> None:
    pack_dir = tmp_path / "workspaces" / "default" / "evals" / pack_name
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "pack.jsonl").write_text("", encoding="utf-8")


def test_gate_uses_real_model_builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.com")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    _prepare_pack(tmp_path, "sample")

    baseline_path = tmp_path / "baseline.json"
    _write_report(baseline_path, 0.8)

    called: dict[str, object] = {}

    def fake_build_model(settings, *args, **kwargs):  # noqa: ANN001
        called["args"] = args
        called["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(cli, "build_model", fake_build_model)
    monkeypatch.setattr(cli, "build_registry", lambda settings, model: object())
    monkeypatch.setattr(cli, "build_agent", lambda settings, model, registry, runtime=None: object())
    monkeypatch.setattr(cli, "WorkflowEngine", lambda model, registry, runtime=None: SimpleNamespace())

    def fake_run_eval_pack(pack_path, agent, engine, report_path):  # noqa: ANN001
        _write_report(report_path, 0.9)
        return json.loads(report_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(cli, "run_eval_pack", fake_run_eval_pack)

    args = cli.parse_subcommand(
        [
            "gate",
            "run",
            "--pack",
            "sample",
            "--baseline",
            str(baseline_path),
            "--report",
            str(tmp_path / "candidate.json"),
        ]
    )
    cli._handle_subcommand(args)

    assert "use_mock" not in called.get("kwargs", {})
    assert called.get("args") == ()


def test_gate_invalid_report_schema_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.com")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    _prepare_pack(tmp_path, "sample")

    baseline_path = tmp_path / "baseline.json"
    _write_report(baseline_path, 0.8)

    monkeypatch.setattr(cli, "build_model", lambda settings: object())
    monkeypatch.setattr(cli, "build_registry", lambda settings, model: object())
    monkeypatch.setattr(cli, "build_agent", lambda settings, model, registry, runtime=None: object())
    monkeypatch.setattr(cli, "WorkflowEngine", lambda model, registry, runtime=None: SimpleNamespace())

    def fake_run_eval_pack(pack_path, agent, engine, report_path):  # noqa: ANN001
        report = {"report_version": "0.1", "overall_score": 0.7}
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(cli, "run_eval_pack", fake_run_eval_pack)

    args = cli.parse_subcommand(
        [
            "gate",
            "run",
            "--pack",
            "sample",
            "--baseline",
            str(baseline_path),
            "--report",
            str(tmp_path / "candidate.json"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        cli._handle_subcommand(args)
    assert exc.value.code == 2
