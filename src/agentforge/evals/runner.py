"""Eval pack runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from agentforge.agent import Agent
from agentforge.workflows.engine import WorkflowEngine, load_workflow_spec
from agentforge.workflows.engine import _validate_schema


@dataclass
class EvalCaseResult:
    case_id: str
    score: float
    passed: bool
    expected: Any
    actual: Any


def load_eval_pack(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            cases.append(json.loads(line))
    return cases


def _score_case(expected: Any, actual: Any, scoring: str, schema: dict[str, Any] | None = None) -> float:
    if scoring == "exact":
        return 1.0 if expected == actual else 0.0
    if scoring == "contains" and isinstance(actual, str):
        return 1.0 if str(expected) in actual else 0.0
    if scoring == "json_schema" and schema:
        errors = _validate_schema(actual, schema)
        return 1.0 if not errors else 0.0
    return 0.0


def run_eval_pack(
    pack_path: Path,
    agent: Agent,
    workflow_engine: WorkflowEngine,
    report_path: Path,
    default_mode: str | None = None,
) -> dict[str, Any]:
    cases = load_eval_pack(pack_path)
    results: list[EvalCaseResult] = []
    total = 0.0
    for case in cases:
        case_id = case.get("id", "case")
        mode = case.get("mode")
        if mode in (None, "auto"):
            mode = default_mode or "agent"
        scoring = case.get("scoring", "exact")
        expected = case.get("expected_output")
        actual: Any = None
        if mode == "agent":
            result = agent.run(case.get("input", ""))
            actual = result.answer
        elif mode == "workflow":
            spec_path = Path(case.get("workflow"))
            spec = load_workflow_spec(spec_path)
            run_result = workflow_engine.run(spec, case.get("input", {}))
            actual = run_result.outputs
        score = _score_case(expected, actual, scoring, case.get("schema"))
        total += score
        results.append(
            EvalCaseResult(
                case_id=case_id,
                score=score,
                passed=score >= 1.0,
                expected=expected,
                actual=actual,
            )
        )
    overall = total / max(1, len(results))
    passed_cases = len([result for result in results if result.passed])
    failures = [
        {"id": result.case_id, "reason": "Score below threshold"}
        for result in results
        if not result.passed
    ]
    eval_pack_id = pack_path.parent.name
    generated_at = datetime.now(timezone.utc).isoformat()
    git_sha = os.environ.get("GITHUB_SHA")
    report = {
        "report_version": "0.1",
        "generated_at": generated_at,
        "git_sha": git_sha,
        "eval_pack_id": eval_pack_id,
        "eval_pack_name": eval_pack_id,
        "overall_score": overall,
        "total_cases": len(results),
        "passed_cases": passed_cases,
        "failures": failures,
        "cases": [
            {
                "id": result.case_id,
                "score": result.score,
                "passed": result.passed,
                "expected": result.expected,
                "actual": result.actual,
            }
            for result in results
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
