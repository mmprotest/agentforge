"""Deterministic fact extraction heuristics."""

from __future__ import annotations

import json
import re
from typing import Any

MAX_FACTS = 20
MAX_FACT_LEN = 200


def extract_facts(
    tool_name: str,
    tool_output: Any,
    summary_text: str,
    source: str | None = None,
) -> list[str]:
    structured = extract_facts_structured(tool_name, tool_output, summary_text, source=source)
    return _finalize_facts([item["value"] for item in structured])


def extract_facts_structured(
    tool_name: str,
    tool_output: Any,
    summary_text: str,
    source: str | None = None,
) -> list[dict[str, str | None]]:
    facts: list[dict[str, str | None]] = []
    raw_text = _stringify(tool_output)
    combined_text = f"{summary_text}\n{raw_text}"
    facts.extend(_wrap("url", _extract_urls(combined_text), source))
    facts.extend(_wrap("number", _extract_numbers_with_units(combined_text), source))
    facts.extend(_wrap("kv", _extract_key_values(tool_output), source))
    facts.extend(_wrap("snippet", _extract_summary_bullets(summary_text), source))
    return _finalize_structured_facts(facts)


def _wrap(kind: str, values: list[str], source: str | None) -> list[dict[str, str | None]]:
    return [{"kind": kind, "value": value, "source": source} for value in values]


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s\"'<>]+", text)


def _extract_numbers_with_units(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\s*(?:[a-zA-Z/%°]+)\b", text)


def _extract_key_values(tool_output: Any) -> list[str]:
    facts: list[str] = []
    if isinstance(tool_output, dict):
        for key, value in tool_output.items():
            if isinstance(value, (str, int, float, bool)) and value is not None:
                facts.append(f"{key}: {value}")
    return facts


def _extract_summary_bullets(summary_text: str) -> list[str]:
    facts: list[str] = []
    for line in summary_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            content = stripped[2:].strip()
            if content:
                facts.append(content)
    return facts


def _finalize_facts(facts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for fact in facts:
        cleaned = fact.strip()
        if not cleaned:
            continue
        if len(cleaned) > MAX_FACT_LEN:
            cleaned = cleaned[: MAX_FACT_LEN - 3] + "..."
        if cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= MAX_FACTS:
            break
    return result


def _finalize_structured_facts(
    facts: list[dict[str, str | None]],
) -> list[dict[str, str | None]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str | None]] = []
    for fact in facts:
        value = str(fact.get("value") or "").strip()
        if not value:
            continue
        if len(value) > MAX_FACT_LEN:
            value = value[: MAX_FACT_LEN - 3] + "..."
        kind = str(fact.get("kind") or "snippet")
        key = (kind, value)
        if key in seen:
            continue
        seen.add(key)
        result.append({"kind": kind, "value": value, "source": fact.get("source")})
        if len(result) >= MAX_FACTS:
            break
    return result
