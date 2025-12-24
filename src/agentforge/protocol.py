"""Structured output protocol for small models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentforge.util.json_repair import JsonRepairError, repair_json


@dataclass(frozen=True)
class ProtocolToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ProtocolFinal:
    answer: str
    confidence: float
    checks: list[str]


def parse_protocol(content: str) -> ProtocolToolCall | ProtocolFinal | None:
    """Parse a protocol object from model content."""
    try:
        payload = repair_json(content)
    except JsonRepairError:
        return None
    if not isinstance(payload, dict):
        return None
    payload_type = payload.get("type")
    if payload_type == "tool":
        name = payload.get("name") or payload.get("tool")
        arguments = payload.get("arguments") or {}
        if isinstance(name, str) and isinstance(arguments, dict):
            return ProtocolToolCall(name=name, arguments=arguments)
        return None
    if payload_type == "final":
        answer = payload.get("answer")
        confidence = payload.get("confidence", 0.0)
        checks = payload.get("checks") or []
        if not isinstance(answer, str):
            return None
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        if not isinstance(checks, list):
            checks = []
        return ProtocolFinal(answer=answer, confidence=float(confidence), checks=[str(c) for c in checks])
    if "name" in payload and "arguments" in payload:
        name = payload.get("name")
        arguments = payload.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            return ProtocolToolCall(name=name, arguments=arguments)
    if "tool" in payload and "arguments" in payload:
        name = payload.get("tool")
        arguments = payload.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            return ProtocolToolCall(name=name, arguments=arguments)
    return None


def format_final(answer: str, checks: list[str]) -> str:
    """Format final user-facing response."""
    summary = "; ".join(checks) if checks else "Answered the request."
    return f"{answer}\n\nWhat I did: {summary}"
