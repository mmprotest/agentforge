from pathlib import Path
import json

import pytest

from agentforge.evals.gating import compare_reports, enforce_min_score, load_report


def test_compare_reports_candidate_better_passes() -> None:
    baseline = {"overall_score": 0.5}
    candidate = {"overall_score": 0.7}
    passed, summary = compare_reports(baseline, candidate, allow_regression=False)
    assert passed is True
    assert summary["delta"] == pytest.approx(0.2)


def test_compare_reports_candidate_worse_fails_without_regression() -> None:
    baseline = {"overall_score": 0.8}
    candidate = {"overall_score": 0.6}
    passed, summary = compare_reports(baseline, candidate, allow_regression=False)
    assert passed is False
    assert summary["delta"] == pytest.approx(-0.2)


def test_missing_baseline_uses_min_score(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"overall_score": 0.4}), encoding="utf-8")
    candidate = load_report(report_path)
    passed, summary = enforce_min_score(candidate, min_score=0.5)
    assert passed is False
    assert summary["candidate_score"] == 0.4
