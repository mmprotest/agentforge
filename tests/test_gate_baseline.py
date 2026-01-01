from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentforge.gate import commands as gate_commands


def _write_config(path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    data = {
        "version": "v1",
        "eval_packs": [{"id": "sample", "path": "evals/sample/pack.jsonl"}],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_baseline_update_writes_report_and_diff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "gate.yml"
    _write_config(config_path)
    report_path = tmp_path / "baseline_report.json"
    diff_path = tmp_path / "baseline_diff.md"

    def fake_run_gate_packs(packs, agent, engine):  # noqa: ANN001
        return {
            "report_version": "0.1",
            "overall_score": 0.5,
            "total_cases": 2,
            "passed_cases": 1,
            "failures": [{"id": "case-1", "reason": "Score below threshold"}],
            "cases": [{"id": "case-1", "expected": "secret", "actual": "secret"}],
            "packs": [],
        }

    monkeypatch.setattr(gate_commands, "run_gate_packs", fake_run_gate_packs)

    gate_commands.baseline_update(
        config_path,
        report_path,
        diff_path,
        None,
        settings=SimpleNamespace(),
        runtime=SimpleNamespace(),
        build_model=lambda *_: object(),
        build_registry=lambda *_: object(),
        build_agent=lambda *_: object(),
        workflow_engine_cls=lambda *_: object(),
    )

    assert report_path.exists()
    diff = diff_path.read_text(encoding="utf-8")
    assert "Baseline Diff" in diff
    assert "No previous baseline provided" in diff


def test_baseline_diff_hides_payloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "gate.yml"
    _write_config(config_path)
    report_path = tmp_path / "baseline_report.json"
    diff_path = tmp_path / "baseline_diff.md"
    prev_path = tmp_path / "prev.json"
    prev_path.write_text(
        json.dumps(
            {
                "report_version": "0.1",
                "overall_score": 0.4,
                "total_cases": 1,
                "passed_cases": 0,
                "failures": [{"id": "case-2", "reason": "Score below threshold"}],
            }
        ),
        encoding="utf-8",
    )

    def fake_run_gate_packs(packs, agent, engine):  # noqa: ANN001
        return {
            "report_version": "0.1",
            "overall_score": 0.5,
            "total_cases": 1,
            "passed_cases": 0,
            "failures": [{"id": "case-1", "reason": "Score below threshold"}],
            "cases": [{"id": "case-1", "expected": "payload", "actual": "payload"}],
            "packs": [],
        }

    monkeypatch.setattr(gate_commands, "run_gate_packs", fake_run_gate_packs)

    gate_commands.baseline_update(
        config_path,
        report_path,
        diff_path,
        prev_path,
        settings=SimpleNamespace(),
        runtime=SimpleNamespace(),
        build_model=lambda *_: object(),
        build_registry=lambda *_: object(),
        build_agent=lambda *_: object(),
        workflow_engine_cls=lambda *_: object(),
    )

    diff = diff_path.read_text(encoding="utf-8")
    assert "payload" not in diff
