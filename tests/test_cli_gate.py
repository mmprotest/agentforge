from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentforge import cli
from agentforge.gate import commands as gate_commands


def _write_report(path: Path, score: float) -> None:
    report = {
        "report_version": "0.1",
        "overall_score": score,
        "failures": [],
        "total_cases": 1,
        "passed_cases": 1,
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def _write_config(path: Path, pack_path: Path, baseline_path: Path) -> None:
    data = {
        "version": "v1",
        "eval_packs": [{"id": "sample", "path": str(pack_path)}],
        "baseline": {"artifact_path": str(baseline_path)},
    }
    yaml = pytest.importorskip("yaml")
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_gate_uses_real_model_builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.com")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    pack_path = tmp_path / "pack.jsonl"
    pack_path.write_text("", encoding="utf-8")

    baseline_path = tmp_path / "baseline.json"
    _write_report(baseline_path, 0.8)
    config_path = tmp_path / "gate.yml"
    _write_config(config_path, pack_path, baseline_path)

    called: dict[str, object] = {}

    def fake_build_model(settings, *args, **kwargs):  # noqa: ANN001
        called["args"] = args
        called["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(cli, "build_model", fake_build_model)
    monkeypatch.setattr(cli, "build_registry", lambda settings, model: object())
    monkeypatch.setattr(cli, "build_agent", lambda settings, model, registry, runtime=None: object())
    monkeypatch.setattr(cli, "WorkflowEngine", lambda model, registry, runtime=None: SimpleNamespace())

    def fake_run_gate_packs(packs, agent, engine):  # noqa: ANN001
        return {
            "report_version": "0.1",
            "overall_score": 0.9,
            "total_cases": 1,
            "passed_cases": 1,
            "failures": [],
            "packs": [],
        }

    monkeypatch.setattr(gate_commands, "run_gate_packs", fake_run_gate_packs)

    args = cli.parse_subcommand(
        [
            "gate",
            "check",
            "--config",
            str(config_path),
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
    pack_path = tmp_path / "pack.jsonl"
    pack_path.write_text("", encoding="utf-8")

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"report_version": "0.1", "overall_score": 0.8}), encoding="utf-8"
    )
    config_path = tmp_path / "gate.yml"
    _write_config(config_path, pack_path, baseline_path)

    monkeypatch.setattr(cli, "build_model", lambda settings: object())
    monkeypatch.setattr(cli, "build_registry", lambda settings, model: object())
    monkeypatch.setattr(cli, "build_agent", lambda settings, model, registry, runtime=None: object())
    monkeypatch.setattr(cli, "WorkflowEngine", lambda model, registry, runtime=None: SimpleNamespace())

    def fake_run_gate_packs(packs, agent, engine):  # noqa: ANN001
        return {
            "report_version": "0.1",
            "overall_score": 0.7,
            "total_cases": 1,
            "passed_cases": 1,
            "failures": [],
            "packs": [],
        }

    monkeypatch.setattr(gate_commands, "run_gate_packs", fake_run_gate_packs)

    args = cli.parse_subcommand(
        [
            "gate",
            "check",
            "--config",
            str(config_path),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        cli._handle_subcommand(args)
    assert exc.value.code == 2
