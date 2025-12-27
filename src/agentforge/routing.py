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
_CODE_RE = re.compile(
    r"```|"
    r"\bpytest\b|"
    r"\btraceback\b|"
    r"\bstack trace\b|"
    r"\bexception\b|"
    r"\bpython\b.*\b(file|script)\b|"
    r"\.py\b",
    re.IGNORECASE,
)
_CODE_TASK_RE = re.compile(
    r"```|"
    r"\bwrite\s+code\b|"
    r"\bimplement\b|"
    r"\bbug\b|"
    r"\btests?\b|"
    r"\btraceback\b|"
    r"\bstack trace\b|"
    r"\bpython\s+function\b",
    re.IGNORECASE,
)
_FILE_RE = re.compile(
    r"(?:^|[\s\"'])((?:\.\.?/|~/)[^\s]+)|"
    r"\b[\w\-. ]+\.(?:py|md|txt|json|yaml|yml|toml|csv|ini|log)\b",
    re.IGNORECASE,
)
_TOOL_REQUEST_RE = re.compile(r"\buse\s+\w+\s+tool\b|\btool\b", re.IGNORECASE)


def tool_candidates(query: str) -> list[RouteSuggestion]:
    normalized = query.strip()
    if not normalized:
        return []
    candidates: list[RouteSuggestion] = []
    if _URL_RE.search(normalized):
        candidates.append(
            RouteSuggestion(tool_name="http_fetch", confidence=0.9, reason="URL detected")
        )
    if _JSON_RE.search(normalized):
        candidates.append(
            RouteSuggestion(tool_name="json_repair", confidence=0.75, reason="JSON parsing requested")
        )
    if _REGEX_RE.search(normalized):
        candidates.append(
            RouteSuggestion(tool_name="regex_extract", confidence=0.7, reason="Regex extraction requested")
        )
    if _CONVERT_RE.search(normalized):
        candidates.append(
            RouteSuggestion(tool_name="unit_convert", confidence=0.7, reason="Unit conversion requested")
        )
    if _ARITH_RE.search(normalized):
        candidates.append(
            RouteSuggestion(tool_name="calculator", confidence=0.8, reason="Arithmetic expression detected")
        )
    if _CODE_RE.search(normalized):
        candidates.append(
            RouteSuggestion(tool_name="code_run_multi", confidence=0.6, reason="Code execution requested")
        )
    if _FILE_RE.search(normalized):
        candidates.append(
            RouteSuggestion(tool_name="filesystem", confidence=0.6, reason="Filesystem operation requested")
        )
    return candidates


def suggest_tool(query: str) -> RouteSuggestion | None:
    candidates = tool_candidates(query)
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.confidence)


def is_code_task(query: str) -> bool:
    normalized = query.strip()
    if not normalized:
        return False
    return bool(_CODE_TASK_RE.search(normalized))


def should_enable_tools(query: str) -> tuple[bool, str]:
    normalized = query.strip()
    if not normalized:
        return False, "empty"
    if _URL_RE.search(normalized):
        return True, "url"
    if _FILE_RE.search(normalized):
        return True, "file"
    if _TOOL_REQUEST_RE.search(normalized):
        return True, "explicit_tool_request"
    if _CODE_RE.search(normalized):
        return True, "code"
    if _ARITH_RE.search(normalized):
        return True, "math"
    if _CONVERT_RE.search(normalized):
        return True, "unit"
    if _JSON_RE.search(normalized):
        return True, "json"
    if _REGEX_RE.search(normalized):
        return True, "regex"
    word_count = len(re.findall(r"\w+", normalized))
    if word_count <= 12:
        return False, "short_closed_book"
    return False, "no_signal"
