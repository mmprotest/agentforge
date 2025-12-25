"""Deterministic controller for agent decisions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from agentforge.policy_engine import PolicyEngine
from agentforge.planner import safe_plan_to_graph
from agentforge.profiles import infer_profile
from agentforge.routing import is_code_task
from agentforge.tasks import CheckSpec, TaskGraph


class ActionType(str, Enum):
    ROUTE_TOOL = "route_tool"
    MODEL_TOOL = "model_tool"
    VERIFY = "verify"
    BACKTRACK = "backtrack"
    FINAL = "final"


class Action(BaseModel):
    type: ActionType
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    message_to_model: str | None = None
    reason: str


class Controller:
    """Deterministic controller that decides next action."""

    def __init__(self, policy_engine: PolicyEngine) -> None:
        self.policy_engine = policy_engine

    def decide(self, state: "AgentState") -> Action:
        if state.task_graph is None or not state.task_graph.tasks:
            state.task_graph = self._initial_task_graph(
                state.query,
                state.profile,
                state.profile_explicit,
                bool(state.memory_state.get("code_check_enabled")),
            )
            state.task_history.append(state.task_graph.model_copy(deep=True))

        if state.task_graph.all_done():
            return Action(type=ActionType.FINAL, reason="tasks complete")

        if state.budgets.model_calls <= 0 and state.budgets.tool_calls <= 0:
            return Action(type=ActionType.FINAL, reason="budget exhausted")

        if state.memory_state.get("pending_verification") and state.budgets.verifies > 0:
            return Action(type=ActionType.VERIFY, reason="verification pending")

        if self._should_backtrack(state):
            return Action(type=ActionType.BACKTRACK, reason="backtrack triggered")

        current_task = state.task_graph.current_task or state.task_graph.next_task()
        if current_task is None:
            return Action(type=ActionType.FINAL, reason="no pending tasks")
        if current_task.id != state.last_snapshot_task_id:
            self._snapshot_state(state)
            state.last_snapshot_task_id = current_task.id
        if (
            current_task.goal.lower().startswith("final")
            and state.memory_state.get("candidate_output") is not None
            and state.budgets.verifies > 0
            and state.memory_state.get("candidate_source") != "tool"
        ):
            return Action(type=ActionType.VERIFY, reason="finalize candidate")
        if (
            self._check_includes_type(current_task.check, "code_run")
            and state.memory_state.get(current_task.inputs.get("source_key"))
            and state.budgets.verifies > 0
            and not state.memory_state.get("needs_revision")
        ):
            return Action(type=ActionType.VERIFY, reason="code check ready")
        if (
            self._check_includes_type(current_task.check, "tool_recompute")
            and state.memory_state.get("last_tool_output") is not None
            and state.budgets.verifies > 0
            and not state.memory_state.get("needs_revision")
        ):
            return Action(type=ActionType.VERIFY, reason="tool recompute ready")

        routing_prompt = state.routing_prompt
        decision = (
            self.policy_engine.route(
                routing_prompt, penalties=state.memory_state.get("route_penalties", {})
            )
            if routing_prompt
            else None
        )
        if decision and decision.must_call and state.budgets.tool_calls > 0:
            return Action(
                type=ActionType.ROUTE_TOOL,
                tool_name=decision.tool_name,
                tool_args=decision.suggested_args,
                reason=decision.reason,
            )

        hint = None
        if decision and decision.suggest_only:
            hint = self.policy_engine.hint_from_decision(decision)

        if state.last_error and state.last_error.get("suggested_fix"):
            hint = self._merge_hint(hint, state.last_error["suggested_fix"])

        return Action(
            type=ActionType.MODEL_TOOL,
            reason="model step",
            message_to_model=hint,
        )

    def _initial_task_graph(
        self,
        query: str,
        profile: str,
        profile_explicit: bool,
        code_check_enabled: bool,
    ) -> TaskGraph:
        if not profile_explicit or profile == "agent":
            profile = infer_profile(query)
        graph = safe_plan_to_graph(query)
        graph.tasks = [self._strengthen_task(task) for task in graph.tasks]
        return graph

    def _strengthen_task(self, task: MicroTask) -> MicroTask:
        check = task.check
        if (
            check.type == "predicate"
            and check.params.get("name") == "non_empty"
            and not check.all_of
            and not check.any_of
        ):
            if task.goal.lower().startswith("final"):
                updated = task.model_copy(deep=True)
                updated.check = CheckSpec(
                    type="contains_fields",
                    params={"required_fields": ["answer"]},
                )
                return updated
            if task.tool_hint:
                updated = task.model_copy(deep=True)
                updated.check = CheckSpec(
                    type="none",
                    all_of=[
                        CheckSpec(type="predicate", params={"name": "non_empty"}),
                        CheckSpec(type="tool_error_absent"),
                    ],
                )
                return updated
        return task

    def _snapshot_state(self, state: "AgentState") -> None:
        from agentforge.state import StateSnapshot

        memory_keys = [
            "facts",
            "facts_structured",
            "constraints",
            "intermediates",
            "draft_answer",
            "candidate_output",
            "candidate_source",
            "candidate_checks",
            "candidate_confidence",
            "pending_verification",
            "verifier_failures",
            "tool_error_counts",
            "last_tool_output",
            "last_tool_name",
            "last_tool_args",
            "final_answer",
            "needs_revision",
            "tool_handles",
            "tool_handle_count",
            "last_tool_facts",
        ]
        subset = {
            key: state.memory_state.get(key)
            for key in memory_keys
            if key in state.memory_state
        }
        snapshot = StateSnapshot(
            task_graph=state.task_graph.model_copy(deep=True),
            memory_state_subset=subset,
            progress_state=state.progress.to_dict(),
            last_tool_summary=state.last_tool_summary,
            last_error=state.last_error,
            tool_error_counts=state.memory_state.get("tool_error_counts", {}),
            verifier_failures=state.memory_state.get("verifier_failures", {}),
            tool_handle_count=int(state.memory_state.get("tool_handle_count", 0) or 0),
            tool_handles=list(state.memory_state.get("tool_handles", []) or []),
        )
        state.snapshots.append(snapshot)

    def _should_backtrack(self, state: "AgentState") -> bool:
        if state.budgets.backtracks <= 0:
            return False
        failures = state.memory_state.get("verifier_failures", {})
        current_task = state.task_graph.current_task
        if current_task and failures.get(current_task.id, 0) >= 2:
            return True
        tool_errors = state.memory_state.get("tool_error_counts", {})
        if any(count >= 2 for count in tool_errors.values()):
            return True
        if state.progress.iterations_without_progress >= 3:
            return True
        return False

    def _merge_hint(self, hint: str | None, fix: str) -> str:
        if hint:
            return f"{hint}\n[Fix] {fix}"
        return f"[Fix] {fix}"

    def _check_includes_type(self, check: CheckSpec, check_type: str) -> bool:
        if check.type == check_type:
            return True
        if check.all_of and any(
            self._check_includes_type(sub, check_type) for sub in check.all_of
        ):
            return True
        if check.any_of and any(
            self._check_includes_type(sub, check_type) for sub in check.any_of
        ):
            return True
        return False
