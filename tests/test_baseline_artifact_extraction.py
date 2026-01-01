from pathlib import Path
import zipfile

import pytest

from agentforge.evals import action as gate_action


def _write_zip(zip_path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(zip_path, "w") as handle:
        for name, content in members.items():
            handle.writestr(name, content)


def test_extract_single_file_exact_match(tmp_path: Path) -> None:
    zip_path = tmp_path / "artifact.zip"
    _write_zip(zip_path, {"baseline_report.json": b"baseline"})

    out_dir = tmp_path / "out"
    extracted = gate_action.extract_single_file_from_zip(
        zip_path, "baseline_report.json", out_dir
    )

    assert extracted.exists()
    assert extracted.read_bytes() == b"baseline"


def test_extract_single_file_missing_expected(tmp_path: Path) -> None:
    zip_path = tmp_path / "artifact.zip"
    _write_zip(zip_path, {"other.json": b"baseline"})

    with pytest.raises(gate_action.BaselineError) as exc:
        gate_action.extract_single_file_from_zip(
            zip_path, "baseline_report.json", tmp_path / "out"
        )

    assert "missing expected file" in str(exc.value)


def test_extract_single_file_blocks_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "artifact.zip"
    _write_zip(zip_path, {"../evil.txt": b"nope"})

    with pytest.raises(gate_action.BaselineError) as exc:
        gate_action.extract_single_file_from_zip(zip_path, "../evil.txt", tmp_path / "out")

    assert "path traversal" in str(exc.value)


def test_extract_single_file_no_fallback_search(tmp_path: Path) -> None:
    zip_path = tmp_path / "artifact.zip"
    _write_zip(zip_path, {"nested/baseline_report.json": b"baseline"})

    with pytest.raises(gate_action.BaselineError) as exc:
        gate_action.extract_single_file_from_zip(
            zip_path, "baseline_report.json", tmp_path / "out"
        )

    assert "missing expected file" in str(exc.value)
