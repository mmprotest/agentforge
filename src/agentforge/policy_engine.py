"""Policy engine for routing and budget control."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from agentforge.profiles import ProfileConfig
from agentforge.routing import RouteSuggestion, suggest_tools
from agentforge.util.logging import get_logger


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
        self.logger = get_logger("agentforge.policy_engine")

    def route(
        self,
        routing_prompt: str,
        penalties: dict[str, int] | None = None,
        disabled_tools: Iterable[str] | None = None,
    ) -> RouteDecision | None:
        suggestions = suggest_tools(routing_prompt)
        if not suggestions:
            return None
        penalties = penalties or {}
        disabled_set = {tool.lower() for tool in (disabled_tools or [])}
        thresholds = self.profile.routing_thresholds
        adjusted: list[RouteSuggestion] = []
        blocked: list[str] = []
        candidates: list[str] = []
        for suggestion in suggestions:
            penalty = penalties.get(suggestion.tool_name, 0)
            confidence = max(0.0, suggestion.confidence - 0.15 * penalty)
            candidates.append(f"{suggestion.tool_name}:{confidence:.2f}")
            # Hard-disabled tools are excluded unless explicitly named by the user.
            if self._is_disabled(suggestion.tool_name, disabled_set, routing_prompt):
                blocked.append(suggestion.tool_name)
                continue
            adjusted.append(
                RouteSuggestion(
                    tool_name=suggestion.tool_name,
                    confidence=confidence,
                    reason=suggestion.reason,
                    suggested_args=suggestion.suggested_args,
                )
            )
        self.logger.info("route.candidates tools=%s", ",".join(candidates))
        if blocked:
            self.logger.info(
                "route.blocked disabled=%s",
                ",".join(sorted(set(blocked))),
            )
        if not adjusted:
            self.logger.info("route.selected tool=None blocked=True")
            return None
        best = max(adjusted, key=lambda suggestion: suggestion.confidence)
        if best.confidence < thresholds.suggest:
            self.logger.info("route.selected tool=None blocked=False")
            return None
        must_call = best.confidence >= thresholds.must_call
        suggest_only = thresholds.suggest <= best.confidence < thresholds.must_call
        self.logger.info(
            "route.selected tool=%s score=%.2f must_call=%s blocked=False",
            best.tool_name,
            best.confidence,
            must_call,
        )
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

    def _is_disabled(
        self, tool_name: str, disabled_set: set[str], routing_prompt: str
    ) -> bool:
        if tool_name.lower() not in disabled_set:
            return False
        # Allow explicitly named tools even if disabled.
        if re.search(rf"\b{re.escape(tool_name)}\b", routing_prompt, re.IGNORECASE):
            return False
        return True
