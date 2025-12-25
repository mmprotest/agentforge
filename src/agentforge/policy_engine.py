"""Policy engine for routing and budget control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentforge.profiles import ProfileConfig
from agentforge.routing import RouteSuggestion, suggest_tools


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

    def route(
        self, routing_prompt: str, penalties: dict[str, int] | None = None
    ) -> RouteDecision | None:
        suggestions = suggest_tools(routing_prompt)
        if not suggestions:
            return None
        penalties = penalties or {}
        thresholds = self.profile.routing_thresholds
        adjusted: list[RouteSuggestion] = []
        for suggestion in suggestions:
            penalty = penalties.get(suggestion.tool_name, 0)
            confidence = max(0.0, suggestion.confidence - 0.15 * penalty)
            adjusted.append(
                RouteSuggestion(
                    tool_name=suggestion.tool_name,
                    confidence=confidence,
                    reason=suggestion.reason,
                    suggested_args=suggestion.suggested_args,
                )
            )
        best = max(adjusted, key=lambda suggestion: suggestion.confidence)
        if best.confidence < thresholds.suggest:
            return None
        must_call = best.confidence >= thresholds.must_call
        suggest_only = thresholds.suggest <= best.confidence < thresholds.must_call
        return RouteDecision(
            tool_name=best.tool_name,
            confidence=best.confidence,
            reason=best.reason,
            suggested_args=best.suggested_args,
            must_call=must_call,
            suggest_only=suggest_only,
        )

    def hint_from_decision(self, decision: RouteDecision) -> str:
        return (
            "[Hint] Router hint: consider using "
            f"{decision.tool_name} ({decision.reason})."
        )
