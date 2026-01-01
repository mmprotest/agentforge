from __future__ import annotations

from pathlib import Path

import pytest

from agentforge.gate import config as gate_config


def test_gate_config_requires_yaml_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "gate.yml"
    config_path.write_text("version: v1\neval_packs:\n  - id: sample\n    path: evals/sample.jsonl\n")

    def _raise() -> None:
        raise gate_config.GateConfigError("Install agentforge[yaml] to use gate.yml.")

    monkeypatch.setattr(gate_config, "_import_yaml", lambda: (_raise() or None))
    with pytest.raises(gate_config.GateConfigError) as exc:
        gate_config.load_gate_config(config_path)
    assert "agentforge[yaml]" in str(exc.value)


def test_gate_config_loads_valid_config(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    config_path = tmp_path / "gate.yml"
    config_path.write_text(
        "\n".join(
            [
                "version: \"v1\"",
                "eval_packs:",
                "  - id: sample",
                "    path: evals/sample.jsonl",
            ]
        ),
        encoding="utf-8",
    )
    config = gate_config.load_gate_config(config_path)
    assert config.version == "v1"
    assert config.eval_packs[0].id == "sample"


def test_gate_config_requires_eval_packs(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    config_path = tmp_path / "gate.yml"
    config_path.write_text("version: \"v1\"\n", encoding="utf-8")
    with pytest.raises(gate_config.GateConfigError):
        gate_config.load_gate_config(config_path)


def test_gate_config_strict_flags_enforced(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    config_path = tmp_path / "gate.yml"
    config_path.write_text(
        "\n".join(
            [
                "version: \"v1\"",
                "eval_packs:",
                "  - id: sample",
                "    path: evals/sample.jsonl",
                "strict:",
                "  require_baseline: false",
                "  no_fallbacks: true",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(gate_config.GateConfigError) as exc:
        gate_config.load_gate_config(config_path)
    assert "require_baseline" in str(exc.value)
