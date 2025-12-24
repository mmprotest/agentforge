"""Best-effort JSON repair utilities."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class JsonRepairError(ValueError):
    """Raised when JSON repair fails."""


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_json_block(text: str) -> str:
    start_index = None
    for idx, char in enumerate(text):
        if char in "{[":
            start_index = idx
            break
    if start_index is None:
        raise JsonRepairError("No JSON object or array found")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start_index, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return text[start_index : idx + 1]
    raise JsonRepairError("Unbalanced JSON braces")


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _replace_single_quotes(text: str) -> str:
    return re.sub(r"(?<!\\)'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', text)


def repair_json(text: str) -> Any:
    """Parse JSON with best-effort repairs."""
    stripped = _strip_fences(text)
    block = _extract_json_block(stripped)
    cleaned = _remove_trailing_commas(block)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        pass
    cleaned = _replace_single_quotes(cleaned)
    cleaned = _remove_trailing_commas(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError) as exc:
        raise JsonRepairError(f"Failed to repair JSON: {exc}") from exc
