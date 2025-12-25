"""Profile configuration for controller behavior."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from agentforge.tasks import CheckSpec, MicroTask, TaskGraph


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
    code_check_max_iters_default: int
    routing_thresholds: RoutingThresholds
    allow_model_plan: bool = False


PROFILES: dict[str, ProfileConfig] = {
    "agent": ProfileConfig(
        name="agent",
        budgets=BudgetConfig(model_calls=20, tool_calls=6, backtracks=2, verifies=6),
        strict_json_default=False,
        verify_default=False,
        code_check_default=False,
        code_check_max_iters_default=2,
        routing_thresholds=RoutingThresholds(must_call=0.85, suggest=0.6),
        allow_model_plan=False,
    ),
    "code": ProfileConfig(
        name="code",
        budgets=BudgetConfig(model_calls=30, tool_calls=8, backtracks=4, verifies=6),
        strict_json_default=True,
        verify_default=True,
        code_check_default=True,
        code_check_max_iters_default=3,
        routing_thresholds=RoutingThresholds(must_call=0.8, suggest=0.6),
        allow_model_plan=False,
    ),
    "math": ProfileConfig(
        name="math",
        budgets=BudgetConfig(model_calls=15, tool_calls=6, backtracks=3, verifies=5),
        strict_json_default=False,
        verify_default=True,
        code_check_default=False,
        code_check_max_iters_default=2,
        routing_thresholds=RoutingThresholds(must_call=0.8, suggest=0.6),
        allow_model_plan=False,
    ),
    "qa": ProfileConfig(
        name="qa",
        budgets=BudgetConfig(model_calls=20, tool_calls=4, backtracks=3, verifies=6),
        strict_json_default=False,
        verify_default=True,
        code_check_default=False,
        code_check_max_iters_default=2,
        routing_thresholds=RoutingThresholds(must_call=0.85, suggest=0.6),
        allow_model_plan=False,
    ),
}

_MATH_RE = re.compile(r"[\d\)][\s]*[+\-*/][\s]*[\d\(]")
_MATH_WORDS = re.compile(r"\bcalculate\b|\bcompute\b|\bsum\b|\badd\b|\bmultiply\b|\bdivide\b")
_CONVERT_RE = re.compile(r"\bconvert\b|\bunit\b", re.IGNORECASE)
_QA_RE = re.compile(r"\bsource\b|\bcitation\b|\bcite\b|\bweb\b|\bwebsite\b|\bhttp", re.IGNORECASE)
_CODE_RE = re.compile(
    r"\bpython\b|\bcode\b|\bimplement\b|\bfunction\b|\btests\b|```",
    re.IGNORECASE,
)


def get_profile(name: str | None) -> ProfileConfig:
    if name and name in PROFILES:
        return PROFILES[name]
    return PROFILES["agent"]


def infer_profile(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        return "agent"
    if _CODE_RE.search(normalized):
        return "code"
    if _MATH_RE.search(normalized) or _MATH_WORDS.search(normalized) or _CONVERT_RE.search(normalized):
        return "math"
    if _QA_RE.search(normalized):
        return "qa"
    return "agent"


def build_graph_code(query: str) -> TaskGraph:
    tasks = [
        _task(
            "draft",
            "Draft solution",
            CheckSpec(
                type="none",
                any_of=[
                    CheckSpec(
                        type="contains_fields",
                        params={"required_fields": ["answer"]},
                    ),
                    CheckSpec(type="regex", params={"pattern": r"```"}),
                ],
            ),
        ),
        _task(
            "code_check",
            "Run code-check",
            CheckSpec(type="code_run", params={"source_key": "draft_answer"}),
            inputs={"source_key": "draft_answer"},
            tool_hint="python_sandbox",
        ),
        _task(
            "final",
            "Finalize",
            CheckSpec(
                type="contains_fields",
                params={"required_fields": ["answer"]},
            ),
        ),
    ]
    return TaskGraph(tasks=tasks, current_task_id=None, history=[])


def build_graph_math(query: str) -> TaskGraph:
    tool_hint = "unit_convert" if _CONVERT_RE.search(query) else "calculator"
    tasks = [
        _task(
            "parse",
            "Parse expression/units",
            CheckSpec(
                type="none",
                any_of=[
                    CheckSpec(
                        type="contains_fields",
                        params={"required_fields": ["expression"]},
                    ),
                    CheckSpec(type="predicate", params={"name": "non_empty"}),
                ],
            ),
        ),
        _task(
            "compute",
            "Compute with calculator",
            CheckSpec(type="tool_recompute", params={}),
            tool_hint=tool_hint,
        ),
        _task(
            "final",
            "Finalize",
            CheckSpec(
                type="none",
                all_of=[
                    CheckSpec(
                        type="contains_fields",
                        params={"required_fields": ["answer"]},
                    ),
                    CheckSpec(type="predicate", params={"name": "looks_numeric"}),
                ],
            ),
        ),
    ]
    return TaskGraph(tasks=tasks, current_task_id=None, history=[])


def build_graph_agent(query: str) -> TaskGraph:
    tasks = [
        _task(
            "identify",
            "Identify needed tools",
            CheckSpec(
                type="contains_fields",
                params={"required_fields": ["answer"]},
            ),
        ),
        _task(
            "tool_step",
            "Execute tool steps",
            CheckSpec(
                type="none",
                all_of=[
                    CheckSpec(type="tool_error_absent"),
                    CheckSpec(type="predicate", params={"name": "non_empty"}),
                ],
            ),
            tool_hint="router",
        ),
        _task(
            "synthesize",
            "Synthesize answer",
            CheckSpec(
                type="contains_fields",
                params={"required_fields": ["answer"]},
            ),
        ),
        _task(
            "final",
            "Finalize",
            CheckSpec(
                type="contains_fields",
                params={"required_fields": ["answer"]},
            ),
        ),
    ]
    return TaskGraph(tasks=tasks, current_task_id=None, history=[])


def build_graph_qa(query: str) -> TaskGraph:
    tasks = [
        _task(
            "decide_tools",
            "Decide if tools are needed",
            CheckSpec(
                type="contains_fields",
                params={"required_fields": ["answer"]},
            ),
        ),
        _task(
            "fetch",
            "Fetch sources",
            CheckSpec(
                type="none",
                all_of=[
                    CheckSpec(type="tool_error_absent"),
                    CheckSpec(type="predicate", params={"name": "non_empty"}),
                ],
            ),
            tool_hint="http_fetch",
        ),
        _task(
            "final",
            "Answer with citations",
            CheckSpec(
                type="contains_fields",
                params={"required_fields": ["answer"]},
            ),
        ),
    ]
    return TaskGraph(tasks=tasks, current_task_id=None, history=[])


def _task(
    task_id: str,
    goal: str,
    check: CheckSpec,
    inputs: dict[str, Any] | None = None,
    tool_hint: str | None = None,
) -> MicroTask:
    return MicroTask(
        id=task_id,
        goal=goal,
        inputs=inputs or {},
        expected_schema=None,
        tool_hint=tool_hint,
        check=check,
        status="pending",
        attempts=0,
        max_attempts=2,
    )
