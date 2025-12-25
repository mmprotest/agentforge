"""Tool-first pre-solver detections for common primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_ARITH_RE = re.compile(r"[\d\)][\s]*[+\-*/][\s]*[\d\(]")
_UNIT_LIST = [
    "km",
    "m",
    "cm",
    "mm",
    "kg",
    "g",
    "lb",
    "oz",
    "°c",
    "°f",
    "mph",
    "kph",
    "l",
    "ml",
]
_UNIT_RE = re.compile(rf"(?P<unit>{'|'.join(_UNIT_LIST)})", re.IGNORECASE)
_UNIT_CONVERT_RE = re.compile(
    rf"(?P<value>\d+(?:\.\d+)?)\s*(?P<from>{'|'.join(_UNIT_LIST)})\s*(?:to|in|into)\s*(?P<to>{'|'.join(_UNIT_LIST)})",
    re.IGNORECASE,
)
_REGEX_RE = re.compile(r"/(?P<pattern>.+?)/")
_STRUCTURED_RE = re.compile(r"\btransform\b|\bnormalize\b|\breshape\b|\breformat\b", re.IGNORECASE)


@dataclass(frozen=True)
class PreSolverDecision:
    tool_name: str
    confidence: float
    reason: str
    suggested_args: dict[str, Any] | None = None


def detect_pre_solvers(query: str) -> list[PreSolverDecision]:
    normalized = query.strip()
    if not normalized:
        return []
    decisions: list[PreSolverDecision] = []
    unit_match = _UNIT_CONVERT_RE.search(normalized)
    if unit_match:
        decisions.append(
            PreSolverDecision(
                tool_name="unit_convert",
                confidence=0.95,
                reason="Unit conversion detected",
                suggested_args={
                    "value": float(unit_match.group("value")),
                    "from_unit": unit_match.group("from"),
                    "to_unit": unit_match.group("to"),
                },
            )
        )
    if "convert" in normalized.lower() and _UNIT_RE.search(normalized):
        decisions.append(
            PreSolverDecision(
                tool_name="unit_convert",
                confidence=0.8,
                reason="Explicit convert with units",
            )
        )
    if _ARITH_RE.search(normalized):
        expression = _extract_expression(normalized)
        decisions.append(
            PreSolverDecision(
                tool_name="calculator",
                confidence=0.9,
                reason="Arithmetic expression detected",
                suggested_args={"expression": expression} if expression else None,
            )
        )
    regex_match = _REGEX_RE.search(normalized)
    if regex_match:
        pattern = regex_match.group("pattern")
        text = _extract_quoted_text(normalized)
        suggested_args = {"pattern": pattern, "text": text} if text else {"pattern": pattern}
        decisions.append(
            PreSolverDecision(
                tool_name="regex_extract",
                confidence=0.8,
                reason="Regex pattern detected",
                suggested_args=suggested_args,
            )
        )
    if _STRUCTURED_RE.search(normalized) and any(
        token in normalized.lower() for token in ["json", "csv", "tsv"]
    ):
        decisions.append(
            PreSolverDecision(
                tool_name="python_sandbox",
                confidence=0.7,
                reason="Structured transform requested",
            )
        )
    return decisions


def _extract_expression(text: str) -> str | None:
    matches = re.findall(r"[0-9\.\s\+\-\*\/\(\)]+", text)
    if not matches:
        return None
    expression = max(matches, key=len).strip()
    return expression or None


def _extract_quoted_text(text: str) -> str | None:
    quoted = re.findall(r"(['\"])(.+?)\1", text, re.DOTALL)
    if not quoted:
        return None
    return max((segment for _, segment in quoted), key=len)
