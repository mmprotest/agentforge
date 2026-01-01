from __future__ import annotations

from pathlib import Path

import pytest

from agentforge.gate.commands import init_gate


def test_gate_init_creates_files(tmp_path: Path) -> None:
    init_gate(tmp_path, force=False)

    assert (tmp_path / "gate.yml").exists()
    assert (tmp_path / ".github" / "workflows" / "ai-baseline.yml").exists()
    assert (tmp_path / ".github" / "workflows" / "ai-regression-gate.yml").exists()
    assert (tmp_path / "evals" / "sample" / "pack.jsonl").exists()
    assert (tmp_path / ".gitignore").exists()


def test_gate_init_respects_force(tmp_path: Path) -> None:
    gate_path = tmp_path / "gate.yml"
    gate_path.write_text("version: \"v1\"\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        init_gate(tmp_path, force=False)
    init_gate(tmp_path, force=True)
