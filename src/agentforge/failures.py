"""Failure taxonomy and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FailureTag(str, Enum):
    """Standardized failure categories for tuning and eval."""

    ROUTER_MISFIRE = "ROUTER_MISFIRE"
    TOOL_ERROR = "TOOL_ERROR"
    PARSE_ERROR = "PARSE_ERROR"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    FORMAT_VIOLATION = "FORMAT_VIOLATION"


def verifier_failure_tag(check_type: str) -> str:
    """Build a verifier failure tag for a specific check type."""
    normalized = check_type.upper()
    return f"VERIFIER_FAIL_{normalized}"


@dataclass(frozen=True)
class FailureEvent:
    """Structured failure event for traces and eval."""

    tag: str
    reason: str
    details: dict[str, Any] | None = None
