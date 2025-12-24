"""Profile configuration for controller behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetConfig:
    model_calls: int
    tool_calls: int
    backtracks: int
    verifies: int


@dataclass(frozen=True)
class RoutingThresholds:
    must_call: float
    suggest: float


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    budgets: BudgetConfig
    strict_json_default: bool
    verify_default: bool
    code_check_default: bool
    routing_thresholds: RoutingThresholds
    allow_model_plan: bool = False


PROFILES: dict[str, ProfileConfig] = {
    "agent": ProfileConfig(
        name="agent",
        budgets=BudgetConfig(model_calls=20, tool_calls=6, backtracks=2, verifies=4),
        strict_json_default=False,
        verify_default=True,
        code_check_default=False,
        routing_thresholds=RoutingThresholds(must_call=0.85, suggest=0.6),
        allow_model_plan=False,
    ),
    "code": ProfileConfig(
        name="code",
        budgets=BudgetConfig(model_calls=30, tool_calls=8, backtracks=4, verifies=6),
        strict_json_default=False,
        verify_default=True,
        code_check_default=True,
        routing_thresholds=RoutingThresholds(must_call=0.8, suggest=0.6),
        allow_model_plan=False,
    ),
    "math": ProfileConfig(
        name="math",
        budgets=BudgetConfig(model_calls=15, tool_calls=6, backtracks=3, verifies=5),
        strict_json_default=False,
        verify_default=True,
        code_check_default=False,
        routing_thresholds=RoutingThresholds(must_call=0.8, suggest=0.6),
        allow_model_plan=False,
    ),
    "qa": ProfileConfig(
        name="qa",
        budgets=BudgetConfig(model_calls=20, tool_calls=4, backtracks=3, verifies=4),
        strict_json_default=False,
        verify_default=True,
        code_check_default=False,
        routing_thresholds=RoutingThresholds(must_call=0.85, suggest=0.6),
        allow_model_plan=False,
    ),
}


def get_profile(name: str | None) -> ProfileConfig:
    if name and name in PROFILES:
        return PROFILES[name]
    return PROFILES["agent"]
