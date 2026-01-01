"""Eval gating summaries."""

from __future__ import annotations

from typing import Any

from agentforge.evals.gating import extract_failures, extract_totals


def build_summary(
    candidate_report: dict[str, Any],
    compare_result: dict[str, Any],
    *,
    passed: bool,
    max_failures: int = 5,
) -> dict[str, Any]:
    total_cases, passed_cases = extract_totals(candidate_report)
    failures = extract_failures(candidate_report, limit=max_failures)
    return {
        "passed": passed,
        "candidate_score": compare_result.get("candidate_score"),
        "baseline_score": compare_result.get("baseline_score"),
        "delta_score": compare_result.get("delta"),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failures": failures,
    }


def render_summary(summary: dict[str, Any]) -> str:
    candidate_score = summary.get("candidate_score")
    baseline_score = summary.get("baseline_score")
    delta_score = summary.get("delta_score")
    failures = summary.get("failures", [])
    failure_text = ", ".join(f"`{item}`" for item in failures) if failures else "None"
    baseline_text = (
        f"{baseline_score:.4f}" if isinstance(baseline_score, (int, float)) else "n/a"
    )
    delta_text = f"{delta_score:.4f}" if isinstance(delta_score, (int, float)) else "n/a"
    candidate_text = (
        f"{candidate_score:.4f}" if isinstance(candidate_score, (int, float)) else "n/a"
    )
    status = "PASS" if summary.get("passed") else "FAIL"
    return "\n".join(
        [
            "## AI Regression Gate",
            f"* Result: **{status}**",
            f"* Candidate score: {candidate_text}",
            f"* Baseline score: {baseline_text}",
            f"* Delta score: {delta_text}",
            f"* Total cases: {summary.get('total_cases', 0)}",
            f"* Passed cases: {summary.get('passed_cases', 0)}",
            f"* Top failing IDs: {failure_text}",
        ]
    )
