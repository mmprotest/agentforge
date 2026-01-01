"""Eval report gating helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPORT_VERSION = "0.1"


class ReportVersionError(ValueError):
    """Raised when the report_version is missing or unsupported."""


class ReportSchemaError(ValueError):
    """Raised when the report schema is invalid."""


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    _validate_report_schema(report)
    return report


def _validate_report_version(report: dict[str, Any]) -> None:
    version = report.get("report_version")
    if not version:
        raise ReportVersionError("Report is missing required report_version.")
    if version != REPORT_VERSION:
        raise ReportVersionError(
            f"Unsupported report_version {version!r}. Expected {REPORT_VERSION}."
        )


def _validate_report_schema(report: dict[str, Any]) -> float:
    _validate_report_version(report)
    if "overall_score" not in report:
        raise ReportSchemaError("Report is missing required overall_score.")
    score = report.get("overall_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ReportSchemaError("overall_score must be a number between 0 and 1.")
    score_value = float(score)
    if not 0.0 <= score_value <= 1.0:
        raise ReportSchemaError("overall_score must be between 0 and 1.")
    if "failures" not in report:
        raise ReportSchemaError("Report is missing required failures list.")
    failures = report.get("failures")
    if not isinstance(failures, list):
        raise ReportSchemaError("failures must be a list.")
    return score_value


def extract_score(report: dict[str, Any]) -> float:
    return _validate_report_schema(report)


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_score = extract_score(baseline)
    candidate_score = extract_score(candidate)
    delta = candidate_score - baseline_score
    pass_reason = "no_regression" if candidate_score >= baseline_score else "regression"
    return {
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "delta": delta,
        "pass_reason": pass_reason,
    }


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    allow_regression: bool = False,
) -> tuple[bool, dict[str, Any]]:
    summary = compare(baseline, candidate)
    passed = True if allow_regression else summary["pass_reason"] == "no_regression"
    summary = {
        **summary,
        "allow_regression": allow_regression,
        "passed": passed,
    }
    return passed, summary


def decide_pass(
    compare_result: dict[str, Any],
    min_score: float,
    allow_regression: bool,
    fail_on_missing_baseline: bool,
    baseline_present: bool,
) -> bool:
    candidate_score = float(compare_result.get("candidate_score", 0.0))
    baseline_score = float(compare_result.get("baseline_score", 0.0))
    if not baseline_present:
        if fail_on_missing_baseline:
            return False
        return candidate_score >= min_score
    regression_pass = True if allow_regression else candidate_score >= baseline_score
    return regression_pass and candidate_score >= min_score


def enforce_min_score(candidate: dict[str, Any], min_score: float) -> tuple[bool, dict[str, Any]]:
    candidate_score = extract_score(candidate)
    passed = candidate_score >= min_score
    return passed, {"candidate_score": candidate_score, "min_score": min_score, "passed": passed}


def extract_failures(report: dict[str, Any], limit: int = 5) -> list[str]:
    _validate_report_schema(report)
    failures: list[str] = []
    for item in report.get("failures", []) or []:
        case_id = str(item.get("id", "case"))
        failures.append(case_id)
    if not failures:
        for case in report.get("cases", []) or []:
            if not case.get("passed", True):
                failures.append(str(case.get("id", "case")))
    return failures[:limit]


def extract_totals(report: dict[str, Any]) -> tuple[int, int]:
    total_cases = report.get("total_cases")
    passed_cases = report.get("passed_cases")
    if total_cases is not None and passed_cases is not None:
        return int(total_cases), int(passed_cases)
    cases = report.get("cases", []) or []
    total = len(cases)
    passed = len([case for case in cases if case.get("passed")])
    return total, passed
