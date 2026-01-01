from pathlib import Path
import json

import pytest

from agentforge.evals.gating import (
    ReportSchemaError,
    ReportVersionError,
    compare,
    decide_pass,
    enforce_min_score,
    load_report,
)


def test_compare_candidate_better_passes() -> None:
    baseline = {"report_version": "0.1", "overall_score": 0.5, "failures": []}
    candidate = {"report_version": "0.1", "overall_score": 0.7, "failures": []}
    summary = compare(baseline, candidate)
    assert summary["delta"] == pytest.approx(0.2)
    assert summary["pass_reason"] == "no_regression"


def test_compare_candidate_worse_regression() -> None:
    baseline = {"report_version": "0.1", "overall_score": 0.8, "failures": []}
    candidate = {"report_version": "0.1", "overall_score": 0.6, "failures": []}
    summary = compare(baseline, candidate)
    assert summary["delta"] == pytest.approx(-0.2)
    assert summary["pass_reason"] == "regression"


def test_missing_baseline_uses_min_score(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"report_version": "0.1", "overall_score": 0.4, "failures": []}),
        encoding="utf-8",
    )
    candidate = load_report(report_path)
    passed, summary = enforce_min_score(candidate, min_score=0.5)
    assert passed is False
    assert summary["candidate_score"] == 0.4


def test_decide_pass_enforces_min_score_with_baseline() -> None:
    compare_result = {"baseline_score": 0.2, "candidate_score": 0.4}
    passed = decide_pass(
        compare_result,
        min_score=0.5,
        allow_regression=True,
        fail_on_missing_baseline=True,
        baseline_present=True,
    )
    assert passed is False


def test_decide_pass_regression_control() -> None:
    compare_result = {"baseline_score": 0.7, "candidate_score": 0.6}
    passed = decide_pass(
        compare_result,
        min_score=0.0,
        allow_regression=False,
        fail_on_missing_baseline=True,
        baseline_present=True,
    )
    assert passed is False
    passed = decide_pass(
        compare_result,
        min_score=0.0,
        allow_regression=True,
        fail_on_missing_baseline=True,
        baseline_present=True,
    )
    assert passed is True


def test_decide_pass_missing_baseline_respects_flag() -> None:
    compare_result = {"candidate_score": 0.8}
    passed = decide_pass(
        compare_result,
        min_score=0.2,
        allow_regression=False,
        fail_on_missing_baseline=True,
        baseline_present=False,
    )
    assert passed is False


def test_report_version_mismatch_raises(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"report_version": "0.0", "overall_score": 0.2, "failures": []}),
        encoding="utf-8",
    )
    with pytest.raises(ReportVersionError):
        load_report(report_path)


def test_missing_failures_raises(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"report_version": "0.1", "overall_score": 0.2}), encoding="utf-8"
    )
    with pytest.raises(ReportSchemaError):
        load_report(report_path)


def test_invalid_score_range_raises(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"report_version": "0.1", "overall_score": 1.5, "failures": []}),
        encoding="utf-8",
    )
    with pytest.raises(ReportSchemaError):
        load_report(report_path)
