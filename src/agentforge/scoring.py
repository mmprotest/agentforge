"""Scoring utilities for eval harness."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str | None = None


def score_exact(predicted: str, expected: str) -> ScoreResult:
    passed = predicted == expected
    return ScoreResult(score=1.0 if passed else 0.0, passed=passed, reason=None)


def score_case_insensitive(predicted: str, expected: str) -> ScoreResult:
    passed = predicted.strip().lower() == expected.strip().lower()
    return ScoreResult(score=1.0 if passed else 0.0, passed=passed, reason=None)


def score_numeric_tolerance(
    predicted: str, expected: str, tolerance: float = 1e-6
) -> ScoreResult:
    try:
        pred_val = float(str(predicted).strip())
        exp_val = float(str(expected).strip())
    except (TypeError, ValueError):
        return ScoreResult(score=0.0, passed=False, reason="Non-numeric value")
    delta = abs(pred_val - exp_val)
    passed = delta <= tolerance
    reason = None if passed else f"Delta {delta} exceeds tolerance {tolerance}"
    return ScoreResult(score=1.0 if passed else 0.0, passed=passed, reason=reason)


def score_regex(predicted: str, pattern: str) -> ScoreResult:
    matched = re.search(pattern, predicted, re.DOTALL) is not None
    return ScoreResult(
        score=1.0 if matched else 0.0,
        passed=matched,
        reason=None if matched else f"Did not match /{pattern}/",
    )


def score_json_schema(predicted: str, schema: dict[str, Any]) -> ScoreResult:
    payload = _load_json(predicted)
    if payload is None:
        return ScoreResult(score=0.0, passed=False, reason="Invalid JSON")
    issues = _validate_schema(schema, payload)
    passed = not issues
    reason = None if passed else "; ".join(issues[:3])
    return ScoreResult(score=1.0 if passed else 0.0, passed=passed, reason=reason)


def _load_json(value: str) -> Any | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _validate_schema(schema: dict[str, Any], payload: Any, path: str = "$") -> list[str]:
    issues: list[str] = []
    expected_type = schema.get("type")
    if expected_type:
        if not _matches_type(payload, expected_type):
            issues.append(f"{path} expected type {expected_type}")
            return issues
    if expected_type == "object" or isinstance(payload, dict):
        required = schema.get("required") or []
        for field in required:
            if field not in payload:
                issues.append(f"{path} missing required field '{field}'")
        properties = schema.get("properties") or {}
        if isinstance(payload, dict):
            for field, field_schema in properties.items():
                if field in payload and isinstance(field_schema, dict):
                    issues.extend(
                        _validate_schema(
                            field_schema,
                            payload[field],
                            path=f"{path}.{field}",
                        )
                    )
    if expected_type == "array" and isinstance(payload, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(payload):
                issues.extend(
                    _validate_schema(item_schema, item, path=f"{path}[{idx}]")
                )
    return issues


def _matches_type(payload: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(payload, dict)
    if expected_type == "array":
        return isinstance(payload, list)
    if expected_type == "string":
        return isinstance(payload, str)
    if expected_type == "number":
        return isinstance(payload, (int, float)) and not isinstance(payload, bool)
    if expected_type == "integer":
        return isinstance(payload, int) and not isinstance(payload, bool)
    if expected_type == "boolean":
        return isinstance(payload, bool)
    return False
