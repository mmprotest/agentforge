"""Deterministic controller for agent decisions."""

from __future__ import annotations

from enum import Enum
import re
from typing import Any

from pydantic import BaseModel

from agentforge.policy_engine import PolicyEngine
from agentforge.planner import safe_plan_to_graph
from agentforge.profiles import infer_profile
from agentforge.routing import is_code_task
from agentforge.tasks import CheckSpec, MicroTask, TaskGraph
from agentforge.tools.registry import ToolRegistry
from agentforge.util.logging import get_logger


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

    def __init__(self, policy_engine: PolicyEngine, registry: ToolRegistry) -> None:
        self.policy_engine = policy_engine
        self.registry = registry
        self.logger = get_logger("agentforge.controller")

    def decide(self, state: "AgentState") -> Action:
        self._log_task_activity(state)
        if state.task_graph is None or not state.task_graph.tasks:
            path_type, reason = self._select_path(state.query)
            state.memory_state["path_type"] = path_type
            state.memory_state["path_reason"] = reason
            state.task_graph = self._initial_task_graph(
                state.query,
                state.profile,
                state.profile_explicit,
                bool(state.memory_state.get("code_check_enabled")),
            )
            self.logger.info(
                "controller.path selected=%s reason=%s",
                path_type,
                reason,
            )
            self.logger.info(
                "controller.task_graph built=%s tasks=%s",
                bool(state.task_graph.tasks),
                len(state.task_graph.tasks),
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
        forced_tool = self._valid_tool_hint(current_task.tool_hint)
        if forced_tool:
            retry_counts = state.memory_state.setdefault("forced_tool_attempts", {})
            attempts = retry_counts.get(current_task.id, 0)
            if attempts >= 2:
                state.task_graph.mark_failed(
                    current_task.id, notes="Forced tool retries exceeded"
                )
                self.logger.info(
                    "controller.task tool_hint=%s task_id=%s forced_retry_exceeded",
                    forced_tool,
                    current_task.id,
                )
                return self.decide(state)
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

        if forced_tool:
            retry_counts = state.memory_state.setdefault("forced_tool_attempts", {})
            retry_counts[current_task.id] = retry_counts.get(current_task.id, 0) + 1
            state.memory_state["forced_tool_name"] = forced_tool
            if self._tool_requires_args(forced_tool):
                return Action(
                    type=ActionType.MODEL_TOOL,
                    tool_name=forced_tool,
                    reason="forced tool requires arguments",
                    message_to_model="[ForceTool] Respond with a tool call only.",
                )
            return Action(
                type=ActionType.ROUTE_TOOL,
                tool_name=forced_tool,
                tool_args={},
                reason="forced tool execution",
            )

        routing_prompt = state.routing_prompt
        decision = None
        if (
            routing_prompt
            and state.memory_state.get("path_type") != "fast"
            and (current_task.tool_hint in (None, "router"))
        ):
            decision = self.policy_engine.route(
                routing_prompt,
                penalties=state.memory_state.get("route_penalties", {}),
                disabled_tools=set(state.memory_state.get("disabled_tools", [])),
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
        if self._select_path(query)[0] == "fast":
            graph = self._fast_task_graph(query)
        else:
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

    def _valid_tool_hint(self, tool_hint: str | None) -> str | None:
        if not tool_hint:
            return None
        normalized = tool_hint.strip()
        if not normalized or normalized.lower() == "router":
            return None
        if self.registry.get(normalized) is None:
            return None
        return normalized

    def _tool_requires_args(self, tool_name: str) -> bool:
        tool = self.registry.get(tool_name)
        if tool is None:
            return False
        schema = tool.input_schema.model_json_schema()
        required = schema.get("required") or []
        return bool(required)

    def _select_path(self, query: str) -> tuple[str, str]:
        text = query.strip()
        if not text:
            return "fast", "empty_query"
        if self._is_mcq(text):
            return "fast", "mcq"
        if self._is_simple_math(text):
            return "fast", "simple_math"
        if self._is_slow_task(text):
            return "slow", "complex_or_tool_heavy"
        if self._is_short_factual(text):
            return "fast", "short_factual"
        return "slow", "default"

    def _fast_task_graph(self, query: str) -> TaskGraph:
        check = CheckSpec(type="predicate", params={"name": "non_empty"})
        if self._is_mcq(query):
            check = CheckSpec(type="predicate", params={"name": "mcq_choice"})
        elif self._is_simple_math(query):
            check = CheckSpec(type="predicate", params={"name": "looks_numeric"})
        task = MicroTask(
            id="fast-1",
            goal="final answer",
            inputs={"query": query},
            expected_schema=None,
            tool_hint=None,
            check=check,
            status="pending",
            attempts=0,
            max_attempts=1,
            notes=None,
        )
        return TaskGraph(tasks=[task])

    def _is_mcq(self, text: str) -> bool:
        return bool(
            re.search(r"\b[A-E]\s*[\).\:]", text, re.IGNORECASE)
            or re.search(r"\boption\s+[A-E]\b", text, re.IGNORECASE)
            or re.search(r"\bchoices?\b", text, re.IGNORECASE)
        )

    def _is_simple_math(self, text: str) -> bool:
        return bool(re.search(r"\d[\d\s\+\-\*/\(\)]+", text))

    def _is_short_factual(self, text: str) -> bool:
        tokens = re.findall(r"\w+", text)
        return len(tokens) <= 8 and not is_code_task(text)

    def _is_slow_task(self, text: str) -> bool:
        lowered = text.lower()
        if is_code_task(text):
            return True
        if any(token in lowered for token in ["steps", "multi-step", "plan", "implement", "build", "analyze", "do work"]):
            return True
        return False

    def _log_task_activity(self, state: "AgentState") -> None:
        tool_event_id = state.memory_state.get("tool_event_id")
        last_logged_tool = state.memory_state.get("logged_tool_event_id")
        if tool_event_id is not None and tool_event_id != last_logged_tool:
            task_id = state.memory_state.get("last_executed_task_id")
            tool_hint = state.memory_state.get("last_task_tool_hint")
            tool_name = state.memory_state.get("last_executed_tool")
            self.logger.info(
                "controller.task tool_exec task_id=%s tool_hint=%s tool=%s",
                task_id,
                tool_hint,
                tool_name,
            )
            state.memory_state["logged_tool_event_id"] = tool_event_id
        verify_event_id = state.memory_state.get("verification_event_id")
        last_logged_verify = state.memory_state.get("logged_verification_event_id")
        if verify_event_id is not None and verify_event_id != last_logged_verify:
            task_id = state.memory_state.get("last_verification_task_id")
            result = state.memory_state.get("last_verification_ok")
            self.logger.info(
                "controller.task verify task_id=%s result=%s",
                task_id,
                result,
            )
            state.memory_state["logged_verification_event_id"] = verify_event_id
