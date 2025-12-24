"""Policy engine for routing and budget control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentforge.profiles import ProfileConfig
from agentforge.routing import RouteSuggestion, suggest_tool


@dataclass(frozen=True)
class RouteDecision:
    tool_name: str
    confidence: float
    reason: str
    suggested_args: dict[str, Any] | None
    must_call: bool
    suggest_only: bool


class PolicyEngine:
    """Applies profile settings to routing decisions."""

    def __init__(self, profile: ProfileConfig) -> None:
        self.profile = profile

    def route(self, routing_prompt: str) -> RouteDecision | None:
        suggestion = suggest_tool(routing_prompt)
        if suggestion is None:
            return None
        thresholds = self.profile.routing_thresholds
        must_call = suggestion.confidence >= thresholds.must_call
        suggest_only = thresholds.suggest <= suggestion.confidence < thresholds.must_call
        return RouteDecision(
            tool_name=suggestion.tool_name,
            confidence=suggestion.confidence,
            reason=suggestion.reason,
            suggested_args=suggestion.suggested_args,
            must_call=must_call,
            suggest_only=suggest_only,
        )

    def hint_from_decision(self, decision: RouteDecision) -> str:
        return (
            "[Hint] Router hint: consider using "
            f"{decision.tool_name} ({decision.reason})."
        )
