"""Dynamic task graph planner with strict schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agentforge.pre_solver import detect_pre_solvers
from agentforge.routing import is_code_task
from agentforge.tasks import CheckSpec, MicroTask, TaskGraph

ALLOWED_CHECK_TYPES = {
    "schema",
    "regex",
    "exact",
    "numeric_tolerance",
    "tool_recompute",
    "unit_sanity",
    "code_run",
}


class PlanTask(BaseModel):
    id: str
    goal: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    tool_hint: str | None = None
    checks: list[CheckSpec]
    max_attempts: int = 2
    budget_hint: str = "medium"


class TaskGraphPlan(BaseModel):
    version: str
    objective: str
    tasks: list[PlanTask]
    final_task_id: str


@dataclass(frozen=True)
class PlanValidationResult:
    ok: bool
    errors: list[str]
    depth: int = 0


class Planner:
    """Rule-based planner that emits a validated task graph plan."""

    def __init__(self, max_tasks: int = 8, max_depth: int = 4) -> None:
        self.max_tasks = max_tasks
        self.max_depth = max_depth

    def plan(self, query: str) -> TaskGraphPlan:
        if is_code_task(query):
            return self._plan_code(query)
        decisions = detect_pre_solvers(query)
        if decisions:
            preferred = max(decisions, key=lambda item: item.confidence)
            return self._plan_with_tool(query, preferred.tool_name)
        return self._default_plan(query)

    def validate(self, plan: TaskGraphPlan) -> PlanValidationResult:
        errors: list[str] = []
        if len(plan.tasks) > self.max_tasks:
            errors.append(f"Too many tasks: {len(plan.tasks)} > {self.max_tasks}")
        ids = [task.id for task in plan.tasks]
        if len(set(ids)) != len(ids):
            errors.append("Task ids must be unique")
        if plan.final_task_id not in ids:
            errors.append("final_task_id must reference a task id")
        for task in plan.tasks:
            for check in task.checks:
                if check.type not in ALLOWED_CHECK_TYPES:
                    errors.append(f"Unsupported check type: {check.type}")
        dag_result = self._validate_dag(plan)
        errors.extend(dag_result.errors)
        return PlanValidationResult(ok=not errors, errors=errors, depth=dag_result.depth)

    def to_task_graph(self, plan: TaskGraphPlan) -> TaskGraph:
        order = self._topological_order(plan)
        tasks = [self._to_microtask(plan, task_id) for task_id in order]
        return TaskGraph(tasks=tasks, current_task_id=None, history=[])

    def _plan_with_tool(self, query: str, tool_name: str) -> TaskGraphPlan:
        if tool_name == "calculator":
            return TaskGraphPlan(
                version="1.0",
                objective="Solve arithmetic precisely",
                tasks=[
                    PlanTask(
                        id="extract",
                        goal="Extract arithmetic expression",
                        checks=[
                            CheckSpec(type="regex", params={"pattern": r"\d"})
                        ],
                        max_attempts=2,
                        budget_hint="low",
                    ),
                    PlanTask(
                        id="compute",
                        goal="Compute expression with calculator",
                        tool_hint="calculator",
                        checks=[CheckSpec(type="tool_recompute", params={})],
                        inputs={"depends_on": ["extract"]},
                        max_attempts=2,
                        budget_hint="low",
                    ),
                    PlanTask(
                        id="final",
                        goal="Format numeric answer",
                        checks=[
                            CheckSpec(
                                type="schema",
                                params={
                                    "schema": {
                                        "type": "object",
                                        "required": ["answer"],
                                        "properties": {"answer": {"type": "string"}},
                                    }
                                },
                            )
                        ],
                        inputs={"depends_on": ["compute"]},
                        max_attempts=2,
                        budget_hint="low",
                    ),
                ],
                final_task_id="final",
            )
        if tool_name == "unit_convert":
            return TaskGraphPlan(
                version="1.0",
                objective="Convert units deterministically",
                tasks=[
                    PlanTask(
                        id="extract",
                        goal="Extract conversion parameters",
                        checks=[
                            CheckSpec(
                                type="regex",
                                params={"pattern": r"\d"},
                            )
                        ],
                        max_attempts=2,
                        budget_hint="low",
                    ),
                    PlanTask(
                        id="convert",
                        goal="Convert units with deterministic tool",
                        tool_hint="unit_convert",
                        checks=[
                            CheckSpec(type="tool_recompute", params={}),
                            CheckSpec(type="unit_sanity", params={"unit": ""}),
                        ],
                        inputs={"depends_on": ["extract"]},
                        max_attempts=2,
                        budget_hint="low",
                    ),
                    PlanTask(
                        id="final",
                        goal="Format converted answer",
                        checks=[
                            CheckSpec(
                                type="schema",
                                params={
                                    "schema": {
                                        "type": "object",
                                        "required": ["answer"],
                                        "properties": {"answer": {"type": "string"}},
                                    }
                                },
                            )
                        ],
                        inputs={"depends_on": ["convert"]},
                        max_attempts=2,
                        budget_hint="low",
                    ),
                ],
                final_task_id="final",
            )
        if tool_name == "regex_extract":
            return TaskGraphPlan(
                version="1.0",
                objective="Extract regex matches",
                tasks=[
                    PlanTask(
                        id="extract",
                        goal="Run regex extraction",
                        tool_hint="regex_extract",
                        checks=[CheckSpec(type="tool_recompute", params={})],
                        max_attempts=2,
                        budget_hint="low",
                    ),
                    PlanTask(
                        id="final",
                        goal="Format extracted output",
                        checks=[
                            CheckSpec(
                                type="schema",
                                params={
                                    "schema": {
                                        "type": "object",
                                        "required": ["answer"],
                                        "properties": {"answer": {"type": "string"}},
                                    }
                                },
                            )
                        ],
                        inputs={"depends_on": ["extract"]},
                        max_attempts=2,
                        budget_hint="low",
                    ),
                ],
                final_task_id="final",
            )
        if tool_name == "python_sandbox":
            return TaskGraphPlan(
                version="1.0",
                objective="Run structured transform in sandbox",
                tasks=[
                    PlanTask(
                        id="transform",
                        goal="Apply deterministic transform",
                        tool_hint="python_sandbox",
                        checks=[CheckSpec(type="tool_recompute", params={})],
                        max_attempts=2,
                        budget_hint="medium",
                    ),
                    PlanTask(
                        id="final",
                        goal="Format transformed output",
                        checks=[
                            CheckSpec(
                                type="schema",
                                params={
                                    "schema": {
                                        "type": "object",
                                        "required": ["answer"],
                                        "properties": {"answer": {"type": "string"}},
                                    }
                                },
                            )
                        ],
                        inputs={"depends_on": ["transform"]},
                        max_attempts=2,
                        budget_hint="low",
                    ),
                ],
                final_task_id="final",
            )
        return self._default_plan(query)

    def _default_plan(self, query: str) -> TaskGraphPlan:
        return TaskGraphPlan(
            version="1.0",
            objective=f"Solve: {query[:80]}",
            tasks=[
                PlanTask(
                    id="constraints",
                    goal="Extract constraints",
                    checks=[CheckSpec(type="regex", params={"pattern": r".+"})],
                    max_attempts=2,
                    budget_hint="low",
                ),
                PlanTask(
                    id="solve",
                    goal="Compute or reason about the solution",
                    checks=[CheckSpec(type="regex", params={"pattern": r".+"})],
                    inputs={"depends_on": ["constraints"]},
                    max_attempts=2,
                    budget_hint="medium",
                ),
                PlanTask(
                    id="final",
                    goal="Format final answer exactly",
                    checks=[
                        CheckSpec(
                            type="schema",
                            params={
                                "schema": {
                                    "type": "object",
                                    "required": ["answer"],
                                    "properties": {"answer": {"type": "string"}},
                                }
                            },
                        )
                    ],
                    inputs={"depends_on": ["solve"]},
                    max_attempts=2,
                    budget_hint="low",
                ),
            ],
            final_task_id="final",
        )

    def _plan_code(self, query: str) -> TaskGraphPlan:
        return TaskGraphPlan(
            version="1.0",
            objective=f"Write correct code for: {query[:80]}",
            tasks=[
                PlanTask(
                    id="draft",
                    goal="Draft solution",
                    checks=[CheckSpec(type="regex", params={"pattern": r".+"})],
                    max_attempts=2,
                    budget_hint="medium",
                ),
                PlanTask(
                    id="code_check",
                    goal="Run code checks",
                    tool_hint="python_sandbox",
                    checks=[CheckSpec(type="code_run", params={"source_key": "draft_answer"})],
                    inputs={"source_key": "draft_answer", "depends_on": ["draft"]},
                    max_attempts=2,
                    budget_hint="medium",
                ),
                PlanTask(
                    id="final",
                    goal="Finalize answer",
                    checks=[
                        CheckSpec(
                            type="schema",
                            params={
                                "schema": {
                                    "type": "object",
                                    "required": ["answer"],
                                    "properties": {"answer": {"type": "string"}},
                                }
                            },
                        )
                    ],
                    inputs={"depends_on": ["code_check"]},
                    max_attempts=2,
                    budget_hint="low",
                ),
            ],
            final_task_id="final",
        )

    def _validate_dag(self, plan: TaskGraphPlan) -> PlanValidationResult:
        deps: dict[str, list[str]] = {}
        for task in plan.tasks:
            deps[task.id] = list(task.inputs.get("depends_on", []))
        errors: list[str] = []
        depth = self._compute_depth(deps, errors)
        if depth > self.max_depth:
            errors.append(f"Task graph depth {depth} exceeds max_depth {self.max_depth}")
        return PlanValidationResult(ok=not errors, errors=errors, depth=depth)

    def _compute_depth(self, deps: dict[str, list[str]], errors: list[str]) -> int:
        visiting: set[str] = set()
        visited: dict[str, int] = {}

        def visit(node: str) -> int:
            if node in visited:
                return visited[node]
            if node in visiting:
                errors.append("Task graph contains a cycle")
                return 0
            visiting.add(node)
            depth = 1
            for parent in deps.get(node, []):
                if parent not in deps:
                    errors.append(f"Unknown dependency: {parent}")
                    continue
                depth = max(depth, 1 + visit(parent))
            visiting.remove(node)
            visited[node] = depth
            return depth

        return max((visit(node) for node in deps), default=0)

    def _topological_order(self, plan: TaskGraphPlan) -> list[str]:
        deps: dict[str, list[str]] = {
            task.id: list(task.inputs.get("depends_on", [])) for task in plan.tasks
        }
        order: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visited:
                return
            if node in visiting:
                return
            visiting.add(node)
            for parent in deps.get(node, []):
                visit(parent)
            visiting.remove(node)
            visited.add(node)
            order.append(node)

        for task in plan.tasks:
            visit(task.id)
        return order

    def _to_microtask(self, plan: TaskGraphPlan, task_id: str) -> MicroTask:
        task = next(task for task in plan.tasks if task.id == task_id)
        check = _combine_checks(task.checks)
        return MicroTask(
            id=task.id,
            goal=task.goal,
            inputs=task.inputs,
            expected_schema=task.checks[0].params.get("schema") if task.checks else None,
            tool_hint=task.tool_hint,
            check=check,
            status="pending",
            attempts=0,
            max_attempts=task.max_attempts,
        )


def _combine_checks(checks: list[CheckSpec]) -> CheckSpec:
    if len(checks) == 1:
        return checks[0]
    return CheckSpec(type="none", all_of=checks)


def build_plan(query: str) -> TaskGraphPlan:
    """Helper to build a validated plan or fall back to default."""
    planner = Planner()
    plan = planner.plan(query)
    validation = planner.validate(plan)
    if validation.ok:
        return plan
    return planner._default_plan(query)


def safe_plan_to_graph(query: str) -> TaskGraph:
    """Build a task graph with validation and fallback."""
    planner = Planner()
    plan = planner.plan(query)
    validation = planner.validate(plan)
    if not validation.ok:
        plan = planner._default_plan(query)
    return planner.to_task_graph(plan)


def validate_plan_payload(payload: dict[str, Any]) -> TaskGraphPlan:
    """Validate a raw plan payload and coerce into TaskGraphPlan."""
    try:
        return TaskGraphPlan.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
