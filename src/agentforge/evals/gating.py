"""Eval report gating helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    allow_regression: bool = False,
) -> tuple[bool, dict[str, Any]]:
    baseline_score = float(baseline.get("overall_score", 0.0))
    candidate_score = float(candidate.get("overall_score", 0.0))
    delta = candidate_score - baseline_score
    passed = True if allow_regression else candidate_score >= baseline_score
    summary = {
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "delta": delta,
        "allow_regression": allow_regression,
        "passed": passed,
    }
    return passed, summary


def enforce_min_score(candidate: dict[str, Any], min_score: float) -> tuple[bool, dict[str, Any]]:
    candidate_score = float(candidate.get("overall_score", 0.0))
    passed = candidate_score >= min_score
    return passed, {"candidate_score": candidate_score, "min_score": min_score, "passed": passed}
