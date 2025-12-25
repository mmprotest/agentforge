"""Rules-based tool routing for small models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agentforge.pre_solver import detect_pre_solvers

@dataclass(frozen=True)
class RouteSuggestion:
    tool_name: str
    confidence: float
    reason: str
    suggested_args: dict[str, Any] | None = None


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_ARITH_RE = re.compile(r"[\d\)][\s]*[+\-*/][\s]*[\d\(]")
_CONVERT_RE = re.compile(r"\bconvert\b", re.IGNORECASE)
_JSON_RE = re.compile(r"\bjson\b|\bparse\b|\brepair\b", re.IGNORECASE)
_JSON_FIX_RE = re.compile(r"\bfix json\b|\bparse failed\b", re.IGNORECASE)
_REGEX_RE = re.compile(r"\bregex\b|\bextract\b", re.IGNORECASE)
_CODE_RE = re.compile(r"\bpython\b|\bpytest\b|\bcode\b|\brun\b", re.IGNORECASE)
_CODE_TASK_RE = re.compile(
    r"\bwrite code\b|\bimplement\b|\bbug\b|\btests\b|\bleetcode\b|\bpython function\b|```",
    re.IGNORECASE,
)
_FILE_INTENT_RE = re.compile(
    r"\bopen file\b|\bread file\b|\bwrite file\b|\bsave\b|\bpath\b|\bdirectory\b|\bfolder\b",
    re.IGNORECASE,
)
_PATH_TOKEN_RE = re.compile(r"(/|\\|\.\/|\.\\|[A-Za-z]:\\|\.txt|\.csv|\.json|\.md)")
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
_UNIT_CONVERT_RE = re.compile(
    rf"(?P<value>\d+(?:\.\d+)?)\s*(?P<from>{'|'.join(_UNIT_LIST)})\s*(?:to|in|into)\s*(?P<to>{'|'.join(_UNIT_LIST)})",
    re.IGNORECASE,
)


def suggest_tools(routing_prompt: str) -> list[RouteSuggestion]:
    normalized = routing_prompt.strip()
    if not normalized:
        return []
    user_query = _extract_user_query(normalized)
    suggestions: list[RouteSuggestion] = []
    for decision in detect_pre_solvers(user_query):
        suggestions.append(
            RouteSuggestion(
                tool_name=decision.tool_name,
                confidence=decision.confidence,
                reason=decision.reason,
                suggested_args=decision.suggested_args,
            )
        )
    if _URL_RE.search(user_query):
        url_match = _URL_RE.search(user_query)
        suggested_args = {"url": url_match.group(0)} if url_match else None
        suggestions.append(
            RouteSuggestion(
                tool_name="http_fetch",
                confidence=0.9,
                reason="URL detected",
                suggested_args=suggested_args,
            )
        )
    if _JSON_FIX_RE.search(normalized):
        suggestions.append(
            RouteSuggestion(
                tool_name="json_repair",
                confidence=0.75,
                reason="JSON repair requested",
            )
        )
    if _JSON_RE.search(normalized):
        suggestions.append(
            RouteSuggestion(
                tool_name="json_repair",
                confidence=0.75,
                reason="JSON parsing requested",
            )
        )
    if _REGEX_RE.search(normalized):
        suggestions.append(
            RouteSuggestion(
                tool_name="regex_extract",
                confidence=0.7,
                reason="Regex extraction requested",
            )
        )
    if _CONVERT_RE.search(normalized) or _UNIT_CONVERT_RE.search(normalized):
        suggestions.append(
            RouteSuggestion(
                tool_name="unit_convert",
                confidence=0.7,
                reason="Unit conversion requested",
            )
        )
    if _ARITH_RE.search(normalized):
        suggestions.append(
            RouteSuggestion(
                tool_name="calculator",
                confidence=0.8,
                reason="Arithmetic expression detected",
            )
        )
    if _CODE_RE.search(normalized):
        suggestions.append(
            RouteSuggestion(
                tool_name="code_run_multi",
                confidence=0.6,
                reason="Code execution requested",
            )
        )
    if _is_filesystem_intent(normalized):
        suggestions.append(
            RouteSuggestion(
                tool_name="filesystem",
                confidence=0.65,
                reason="Filesystem operation requested",
            )
        )
    return suggestions


def suggest_tool(routing_prompt: str) -> RouteSuggestion | None:
    suggestions = suggest_tools(routing_prompt)
    if not suggestions:
        return None
    return max(suggestions, key=lambda suggestion: suggestion.confidence)


def _extract_user_query(routing_prompt: str) -> str:
    for line in routing_prompt.splitlines():
        if line.lower().startswith("user query:"):
            return line.split(":", 1)[1].strip()
    return routing_prompt


def is_code_task(query: str) -> bool:
    normalized = query.strip()
    if not normalized:
        return False
    return bool(_CODE_TASK_RE.search(normalized))


def _is_filesystem_intent(text: str) -> bool:
    if _FILE_INTENT_RE.search(text):
        return True
    return bool(_PATH_TOKEN_RE.search(text))
