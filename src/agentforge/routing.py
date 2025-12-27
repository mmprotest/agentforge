"""Rules-based tool routing for small models."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteSuggestion:
    tool_name: str
    confidence: float
    reason: str


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_ARITH_RE = re.compile(r"[\d\)][\s]*[+\-*/][\s]*[\d\(]")
_CONVERT_RE = re.compile(
    r"\bconvert\s+\d+(?:\.\d+)?\s*[a-zA-Z]+\s+to\s+[a-zA-Z]+\b"
    r"|"
    r"\b\d+(?:\.\d+)?\s*[a-zA-Z]+\s+to\s+[a-zA-Z]+\b",
    re.IGNORECASE,
)
_JSON_RE = re.compile(r"\bjson\b|\bparse\b|\brepair\b", re.IGNORECASE)
_REGEX_RE = re.compile(r"\bregex\b|\bextract\b", re.IGNORECASE)
_CODE_RE = re.compile(r"\bpython\b|\bpytest\b|\bcode\b|\brun\b", re.IGNORECASE)
_CODE_TASK_RE = re.compile(
    r"\bwrite code\b|\bimplement\b|\bbug\b|\btests\b|\bleetcode\b|\bpython function\b|```",
    re.IGNORECASE,
)
_FILE_RE = re.compile(r"\bfile\b|\bread\b|\bwrite\b", re.IGNORECASE)


def suggest_tool(query: str) -> RouteSuggestion | None:
    normalized = query.strip()
    if not normalized:
        return None
    if _URL_RE.search(normalized):
        return RouteSuggestion(tool_name="http_fetch", confidence=0.9, reason="URL detected")
    if _JSON_RE.search(normalized):
        return RouteSuggestion(tool_name="json_repair", confidence=0.75, reason="JSON parsing requested")
    if _REGEX_RE.search(normalized):
        return RouteSuggestion(tool_name="regex_extract", confidence=0.7, reason="Regex extraction requested")
    if _CONVERT_RE.search(normalized):
        return RouteSuggestion(tool_name="unit_convert", confidence=0.7, reason="Unit conversion requested")
    if _ARITH_RE.search(normalized):
        return RouteSuggestion(tool_name="calculator", confidence=0.8, reason="Arithmetic expression detected")
    if _CODE_RE.search(normalized):
        return RouteSuggestion(tool_name="code_run_multi", confidence=0.6, reason="Code execution requested")
    if _FILE_RE.search(normalized):
        return RouteSuggestion(tool_name="filesystem", confidence=0.6, reason="Filesystem operation requested")
    return None


def is_code_task(query: str) -> bool:
    normalized = query.strip()
    if not normalized:
        return False
    return bool(_CODE_TASK_RE.search(normalized))
