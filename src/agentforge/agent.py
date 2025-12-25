"""Core agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import sys
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from agentforge.controller import ActionType, Controller
from agentforge.failures import FailureEvent, FailureTag, verifier_failure_tag
from agentforge.models.base import BaseChatModel, ModelResponse
from agentforge.protocol import (
    ProtocolFinal,
    ProtocolToolCall,
    format_final,
    parse_protocol,
    protocol_from_payload,
)
from agentforge.memory import MemoryEntry, MemoryStore
from agentforge.policy_engine import PolicyEngine
from agentforge.profiles import get_profile
from agentforge.routing import is_code_task
from agentforge.safety.policy import SafetyPolicy
from agentforge.state import AgentBudgets, AgentState
from agentforge.tasks import CheckSpec, MicroTask
from agentforge.trace import TraceRecorder
from agentforge.tools.base import Tool
from agentforge.tools.builtins.deep_think import DeepThinkTool
from agentforge.tools.registry import ToolRegistry
from agentforge.util.context_trim import trim_messages
from agentforge.util.fact_extract import extract_facts, extract_facts_structured
from agentforge.util.json_repair import JsonRepairError, repair_json
from agentforge.util.progress import ProgressTracker
from agentforge.verifier import Verifier


@dataclass
class AgentResult:
    answer: str
    tools_used: list[str] = field(default_factory=list)
    tools_created: list[str] = field(default_factory=list)
    trace_path: str | None = None
    checks: list[str] = field(default_factory=list)
    confidence: float | None = None


class Agent:
    """Agent that can call tools and optionally create new tools."""

    def __init__(
        self,
        model: BaseChatModel,
        registry: ToolRegistry,
        policy: SafetyPolicy | None = None,
        mode: str = "direct",
        verify: bool | None = None,
        self_consistency: int = 1,
        max_model_calls: int | None = None,
        memory: MemoryStore | None = None,
        trace: TraceRecorder | None = None,
        strict_json_mode: bool | None = None,
        max_message_chars: int = 24000,
        max_message_tokens_approx: int = 6000,
        token_char_ratio: int = 4,
        max_single_message_chars: int = 4000,
        max_turns: int = 20,
        trim_strategy: str = "drop_oldest",
        code_check: bool | None = None,
        code_check_max_iters: int | None = None,
        profile: str | None = None,
        branch_candidates: int = 1,
        eval_mode: bool = False,
    ) -> None:
        self.model = model
        self.registry = registry
        self.policy = policy or SafetyPolicy()
        self.mode = mode
        self.verify = verify
        self.self_consistency = max(1, min(self_consistency, 3))
        self.max_model_calls = max_model_calls or self.policy.max_model_calls
        self.memory = memory or MemoryStore()
        self.trace = trace
        self.strict_json_mode = strict_json_mode
        self.max_message_chars = max_message_chars
        self.max_message_tokens_approx = max_message_tokens_approx
        self.token_char_ratio = token_char_ratio
        self.max_single_message_chars = max_single_message_chars
        self.max_turns = max_turns
        self.trim_strategy = trim_strategy
        self.code_check = code_check
        self.code_check_max_iters = code_check_max_iters
        self.profile_name = profile or "agent"
        self.profile_explicit = profile is not None
        self.branch_candidates = max(1, min(int(branch_candidates), 3))
        self.eval_mode = bool(eval_mode)
        self.tools_used: list[str] = []
        self.tools_created: list[str] = []
        self._model_calls = 0
        self._last_state: AgentState | None = None

    def _internal_plan(self, query: str) -> None:
        if self.mode != "deep":
            return
        deep_tool = self.registry.get("deep_think")
        if isinstance(deep_tool, DeepThinkTool):
            result = deep_tool.run({"problem": query, "constraints": []})
            plan = result.output
        else:
            plan = {"plan": ["Use available tools"], "checks": []}
        self.memory.state["plan"] = plan
        if self.trace:
            self.trace.record("internal_plan", {"plan": plan})

    def _build_router_prompt(
        self,
        query: str,
        last_tool_summary: str | None,
        last_parse_failed: str | None,
    ) -> str:
        parts = [f"User query: {query}"]
        facts = self.memory.state.get("facts")
        if isinstance(facts, list) and facts:
            parts.append(f"Known facts: {'; '.join(facts)}")
        if last_tool_summary:
            parts.append(f"Last tool summary: {last_tool_summary}")
        if last_parse_failed:
            parts.append("Previous model output failed to parse as JSON.")
        return "\n".join(parts)

    def _append_message(self, message: dict[str, Any]) -> None:
        self._messages.append(message)
        if self.trim_strategy == "drop_oldest":
            self._messages = trim_messages(
                self._messages,
                self.max_message_chars,
                self.max_turns,
                self.max_message_tokens_approx,
                self.token_char_ratio,
                self.max_single_message_chars,
            )

    def run(self, query: str) -> AgentResult:
        profile = get_profile(self.profile_name)
        verify_enabled = profile.verify_default if self.verify is None else self.verify
        if self.eval_mode:
            verify_enabled = True
        if self.self_consistency > 1:
            candidates: list[AgentResult] = []
            for index in range(self.self_consistency):
                if self._model_calls >= self.max_model_calls:
                    break
                nonce = f"run-{index}-{uuid4().hex[:6]}"
                result = self._run_once(query, nonce=nonce)
                candidates.append(result)
            if not candidates:
                return AgentResult(answer="No response from model")
            best = self._pick_best(candidates)
            if verify_enabled and self._model_calls < self.max_model_calls:
                verified = self._verify_answer(query, best.answer)
                best.answer = verified.answer
                best.checks = verified.checks
                best.confidence = verified.confidence
            return best
        result = self._run_once(query)
        if verify_enabled and self._model_calls < self.max_model_calls:
            verified = self._verify_answer(query, result.answer)
            result.answer = verified.answer
            result.checks = verified.checks
            result.confidence = verified.confidence
        return result

    def _run_once(self, query: str, nonce: str | None = None) -> AgentResult:
        self.tools_used = []
        self.tools_created = []
        self.memory = MemoryStore(
            max_tool_output_chars=self.memory.max_tool_output_chars,
            keep_raw_tool_output=self.memory.keep_raw_tool_output,
            summary_lines=self.memory.summary_lines,
        )
        self._messages = []
        profile = get_profile(self.profile_name)
        strict_json_mode = profile.strict_json_default if self.strict_json_mode is None else self.strict_json_mode
        if self.eval_mode:
            strict_json_mode = True
        self.strict_json_mode = strict_json_mode
        effective_code_check_max_iters = (
            profile.code_check_max_iters_default
            if self.code_check_max_iters is None
            else max(1, self.code_check_max_iters)
        )
        self.code_check_max_iters = effective_code_check_max_iters
        system_prompt = (
            "You are an agent that can use tools. "
            "Use tools when necessary. "
            "Respond with JSON only, using either "
            '{"type":"tool","name":"<tool>","arguments":{...}} '
            'or {"type":"final","answer":"...","scratchpad":"...","confidence":0.0,"checks":[...]} '
            "Do not reveal chain-of-thought. Scratchpad is hidden from users."
        )
        if strict_json_mode:
            system_prompt += (
                " Output must be exactly one JSON object and nothing else."
            )
        self._append_message({"role": "system", "content": system_prompt})
        if nonce:
            self._append_message({"role": "system", "content": f"Nonce: {nonce}"})
        self._append_message({"role": "user", "content": query})
        self._internal_plan(query)
        if self.trace:
            self.trace.record_messages(self._messages)

        policy_engine = PolicyEngine(profile)
        controller = Controller(policy_engine, self.registry)
        verifier = Verifier(self._run_tool)
        budgets = AgentBudgets(
            model_calls=min(self.max_model_calls, profile.budgets.model_calls),
            tool_calls=min(self.policy.max_tool_calls, profile.budgets.tool_calls),
            backtracks=profile.budgets.backtracks,
            verifies=profile.budgets.verifies,
        )
        base_code_check = (
            profile.code_check_default
            if self.code_check is None
            else bool(self.code_check)
        )
        code_check_enabled = base_code_check or (
            profile.code_check_default and is_code_task(query)
        )
        state = AgentState(
            query=query,
            task_graph=None,
            memory_state=self.memory.state,
            last_tool_summary=None,
            last_error=None,
            progress=ProgressTracker(),
            budgets=budgets,
            profile=profile.name,
            task_history=[],
            routing_prompt="",
            snapshots=[],
            last_snapshot_task_id=None,
            profile_explicit=self.profile_explicit,
        )
        state.memory_state.update(
            {
                "candidate_output": None,
                "candidate_source": None,
                "candidate_checks": [],
                "candidate_confidence": None,
                "pending_verification": False,
                "verifier_failures": {},
                "tool_error_counts": {},
                "route_penalties": {},
                "last_routed_tool": None,
                "backtrack_count": 0,
                "draft_answer": None,
                "final_answer": None,
                "code_check_enabled": code_check_enabled,
                "needs_revision": False,
                "tool_handles": [],
                "tool_handle_count": 0,
                "last_tool_facts": [],
                "scratchpad": None,
                "fast_path": False,
                "path_type_logged": False,
                "tool_failure_counts": {},
                "disabled_tools": [],
            }
        )
        state.memory_state.setdefault("facts_structured", [])
        state.memory_state.setdefault("constraints", self._extract_constraints_from_query(query))
        state.memory_state.setdefault("intermediates", {})
        last_parse_failed: str | None = None
        retry_instruction: dict[str, Any] | None = None
        format_retry_message: dict[str, Any] | None = None
        format_retry_remaining = 1 if self.strict_json_mode else 0
        for _ in range(self.max_turns):
            state.routing_prompt = self._build_router_prompt(
                query, state.last_tool_summary, last_parse_failed
            )
            action = controller.decide(state)
            if state.memory_state.get("path_type") and not state.memory_state.get("path_type_logged"):
                state.memory_state["fast_path"] = state.memory_state.get("path_type") == "fast"
                if self.trace:
                    self.trace.record(
                        "controller_path",
                        {
                            "path": state.memory_state.get("path_type"),
                            "reason": state.memory_state.get("path_reason"),
                        },
                    )
                state.memory_state["path_type_logged"] = True
            if action.type == ActionType.ROUTE_TOOL:
                if state.budgets.tool_calls <= 0:
                    state.memory_state["pending_verification"] = True
                    continue
                current_task = state.task_graph.current_task
                tool_name = action.tool_name or ""
                state.memory_state["last_routed_tool"] = tool_name
                direct_args = (
                    action.tool_args
                    if action.tool_args is not None
                    else self._direct_tool_args(tool_name, query, last_parse_failed)
                )
                if direct_args is None:
                    state.last_error = {
                        "error": f"Missing arguments for tool {tool_name}",
                        "suggested_fix": "Provide valid tool arguments.",
                    }
                    state.memory_state["pending_verification"] = True
                    continue
                tool = self.registry.get(tool_name)
                if tool is None:
                    self._record_failure(
                        FailureEvent(
                            tag=FailureTag.TOOL_ERROR.value,
                            reason="Requested unknown tool",
                            details={"tool_name": tool_name},
                        )
                    )
                    return AgentResult(
                        answer=f"Requested unknown tool: {tool_name}",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                if not self.policy.is_tool_allowed(tool_name):
                    self._record_failure(
                        FailureEvent(
                            tag=FailureTag.TOOL_ERROR.value,
                            reason="Tool not allowed",
                            details={"tool_name": tool_name},
                        )
                    )
                    return AgentResult(
                        answer=f"Tool not allowed: {tool_name}",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                output, ok = self._execute_tool(tool, direct_args)
                entry = self._handle_tool_result(
                    tool_name, output, None, direct_args
                )
                self._record_tool_execution(state, current_task, tool_name)
                state.budgets.tool_calls -= 1
                state.last_tool_summary = entry.summary
                self.memory.state["last_tool_summary"] = state.last_tool_summary
                state.memory_state["candidate_output"] = output
                state.memory_state["candidate_source"] = "tool"
                state.memory_state["pending_verification"] = True
                state.memory_state["last_tool_output"] = output
                state.memory_state["last_tool_name"] = tool_name
                state.memory_state["last_tool_args"] = direct_args
                last_parse_failed = None
                if not ok:
                    self._record_tool_error(state, tool_name)
                    retry_instruction = self._build_retry_instruction(tool, output)
                continue
            if action.type == ActionType.MODEL_TOOL:
                if state.budgets.model_calls <= 0:
                    state.memory_state["pending_verification"] = True
                    continue
                state.memory_state["last_routed_tool"] = None
                call_messages = list(self._messages)
                contract_message = self._build_contract_message(
                    state,
                    hint=action.message_to_model,
                    retry_instruction=retry_instruction,
                    format_retry_message=format_retry_message,
                    strict_json_mode=strict_json_mode,
                )
                if contract_message:
                    call_messages.append(contract_message)
                retry_instruction = None
                format_retry_message = None
                tools = [] if state.memory_state.get("fast_path") else self.registry.openai_schemas()
                current_task = state.task_graph.current_task
                response = self._run_model_candidates(
                    call_messages,
                    tools,
                    state,
                    current_task,
                    verifier,
                    strict_json_mode,
                )
                protocol = None
                forced_tool_name = action.tool_name
                if response.final_text:
                    protocol = self._parse_model_protocol(response.final_text)
                    if protocol is None and strict_json_mode:
                        if code_check_enabled and self._extract_python_blocks(
                            response.final_text
                        ):
                            state.memory_state["candidate_output"] = response.final_text
                            state.memory_state["candidate_source"] = "model"
                            state.memory_state["pending_verification"] = True
                            last_parse_failed = None
                            continue
                        if format_retry_remaining > 0:
                            format_retry_message = self._build_format_retry_message()
                            format_retry_remaining -= 1
                            last_parse_failed = response.final_text
                            continue
                        last_parse_failed = response.final_text
                        self._record_failure(
                            FailureEvent(
                                tag=FailureTag.FORMAT_VIOLATION.value,
                                reason="Strict JSON output required",
                                details={"output": response.final_text[:200]},
                            )
                        )
                        return AgentResult(
                            answer="Could not parse the model output as JSON.",
                            tools_used=self.tools_used,
                            tools_created=self.tools_created,
                            trace_path=self._finalize_trace(),
                        )
                    if protocol is not None:
                        last_parse_failed = None
                if forced_tool_name:
                    if response.tool_call is None and isinstance(protocol, ProtocolToolCall):
                        response = response.model_copy(
                            update={"tool_call": protocol_to_toolcall(protocol)}
                        )
                    if response.tool_call is None or response.tool_call.name != forced_tool_name:
                        state.last_error = {
                            "error": f"Forced tool call required: {forced_tool_name}",
                            "suggested_fix": f"Return a tool call for {forced_tool_name} only.",
                        }
                        state.memory_state["forced_tool_model_failures"] = (
                            state.memory_state.get("forced_tool_model_failures", 0) + 1
                        )
                        last_parse_failed = response.final_text
                        self._record_failure(
                            FailureEvent(
                                tag=FailureTag.FORMAT_VIOLATION.value,
                                reason="Forced tool call required",
                                details={"tool_name": forced_tool_name},
                            )
                        )
                        continue
                if response.tool_call is None and isinstance(protocol, ProtocolToolCall):
                    response = response.model_copy(
                        update={"tool_call": protocol_to_toolcall(protocol)}
                    )
                if response.final_text is not None and isinstance(protocol, ProtocolFinal):
                    answer = format_final(protocol.answer, protocol.checks, eval_mode=self.eval_mode)
                    if protocol.scratchpad:
                        state.memory_state["scratchpad"] = protocol.scratchpad
                        if self.trace:
                            self.trace.record("scratchpad", {"content": protocol.scratchpad})
                    if code_check_enabled:
                        state.memory_state["draft_answer"] = answer
                    state.memory_state["candidate_output"] = answer
                    state.memory_state["candidate_source"] = "model"
                    state.memory_state["candidate_checks"] = protocol.checks
                    state.memory_state["candidate_confidence"] = protocol.confidence
                    state.memory_state["pending_verification"] = True
                    if current_task and current_task.goal.lower().startswith("final"):
                        state.memory_state["final_answer"] = answer
                    continue
                if response.final_text is not None and response.tool_call is None:
                    answer = response.final_text
                    if code_check_enabled:
                        state.memory_state["draft_answer"] = answer
                    state.memory_state["candidate_output"] = answer
                    state.memory_state["candidate_source"] = "model"
                    state.memory_state["pending_verification"] = True
                    if current_task and current_task.goal.lower().startswith("final"):
                        state.memory_state["final_answer"] = answer
                    continue
                if response.tool_call is None:
                    self._record_failure(
                        FailureEvent(
                            tag=FailureTag.PARSE_ERROR.value,
                            reason="No response from model",
                        )
                    )
                    return AgentResult(
                        answer="No response from model",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                tool_name = response.tool_call.name
                tool = self.registry.get(tool_name)
                if tool is None:
                    self._record_failure(
                        FailureEvent(
                            tag=FailureTag.TOOL_ERROR.value,
                            reason="Requested unknown tool",
                            details={"tool_name": tool_name},
                        )
                    )
                    return AgentResult(
                        answer=f"Requested unknown tool: {tool_name}",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                if not self.policy.is_tool_allowed(tool_name):
                    self._record_failure(
                        FailureEvent(
                            tag=FailureTag.TOOL_ERROR.value,
                            reason="Tool not allowed",
                            details={"tool_name": tool_name},
                        )
                    )
                    return AgentResult(
                        answer=f"Tool not allowed: {tool_name}",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                output, ok = self._execute_tool(tool, response.tool_call.arguments)
                entry = self._handle_tool_result(
                    tool_name,
                    output,
                    response.tool_call.id,
                    response.tool_call.arguments,
                )
                self._record_tool_execution(state, current_task, tool_name)
                state.last_tool_summary = entry.summary
                self.memory.state["last_tool_summary"] = state.last_tool_summary
                state.memory_state["candidate_output"] = output
                state.memory_state["last_tool_output"] = output
                state.memory_state["last_tool_name"] = tool_name
                state.memory_state["last_tool_args"] = response.tool_call.arguments
                state.memory_state["candidate_source"] = "tool"
                state.memory_state["pending_verification"] = True
                if not ok:
                    self._record_tool_error(state, tool_name)
                    retry_instruction = self._build_retry_instruction(tool, output)
                continue
            if action.type == ActionType.VERIFY:
                current_task = state.task_graph.current_task or state.task_graph.next_task()
                if current_task is None:
                    state.memory_state["pending_verification"] = False
                    continue
                candidate = state.memory_state.get("candidate_output")
                if candidate is None and current_task.goal.lower().startswith("final"):
                    candidate = state.memory_state.get("draft_answer")
                candidate_for_check = self._coerce_candidate_for_check(
                    current_task, candidate
                )
                if self._check_includes_type(current_task.check, "code_run"):
                    source_key = current_task.inputs.get("source_key")
                    if candidate is None and source_key:
                        candidate = state.memory_state.get(source_key)
                    if source_key:
                        current_task = self._attach_code_source(current_task, candidate)
                current_task = self._attach_tool_error_absent(current_task, state)
                if self._check_includes_type(current_task.check, "tool_recompute"):
                    current_task = self._attach_tool_recompute(current_task, state)
                result = verifier.verify(candidate_for_check, current_task)
                state.budgets.verifies -= 1
                state.memory_state["pending_verification"] = False
                self._record_verifier_result(state, current_task, result)
                self._record_verification_event(state, current_task, result.passed)
                if result.passed:
                    state.task_graph.mark_done(current_task.id)
                    if current_task.goal.lower().startswith("draft"):
                        state.memory_state["draft_answer"] = candidate
                    if self._check_includes_type(current_task.check, "code_run"):
                        state.memory_state["draft_answer"] = candidate
                    if current_task.goal.lower().startswith("final"):
                        state.memory_state["final_answer"] = candidate
                else:
                    if self._is_forced_tool_task(current_task):
                        state.task_graph.mark_failed(
                            current_task.id, notes="Forced tool verification failed"
                        )
                    else:
                        state.task_graph.record_attempt(current_task.id)
                        if current_task.attempts >= current_task.max_attempts:
                            state.task_graph.mark_failed(
                                current_task.id, notes="Max attempts exceeded"
                            )
                self._update_progress(state, result)
                continue
            if action.type == ActionType.BACKTRACK:
                self._apply_backtrack(state, action.reason)
                continue
            if action.type == ActionType.FINAL:
                answer = state.memory_state.get("final_answer") or state.memory_state.get("draft_answer")
                if not answer and state.memory_state.get("candidate_output"):
                    answer = state.memory_state["candidate_output"]
                if not answer:
                    answer = "No response from model"
                if action.reason == "budget exhausted":
                    self._record_failure(
                        FailureEvent(
                            tag=FailureTag.BUDGET_EXHAUSTED.value,
                            reason="Budget exhausted before completion",
                        )
                    )
                self._last_state = state
                return AgentResult(
                    answer=str(answer).rstrip() if self.eval_mode else str(answer),
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    checks=state.memory_state.get("candidate_checks", []),
                    confidence=state.memory_state.get("candidate_confidence"),
                    trace_path=self._finalize_trace(),
                )
        self._last_state = state
        self._record_failure(
            FailureEvent(
                tag=FailureTag.BUDGET_EXHAUSTED.value,
                reason="Reached iteration limit",
            )
        )
        return AgentResult(
            answer="Reached iteration limit",
            tools_used=self.tools_used,
            tools_created=self.tools_created,
            trace_path=self._finalize_trace(),
        )

    def _handle_tool_result(
        self,
        tool_name: str,
        output: Any,
        call_id: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        if tool_name == "tool_maker":
            tool_created = output.get("tool") if isinstance(output, dict) else None
            if tool_created:
                self.tools_created.append(tool_created)
        entry = self.memory.add_tool_output(tool_name, output)
        self._update_facts_from_tool(tool_name, output, entry.summary)
        self.memory.state["tool_handles"] = [item.handle for item in self.memory.entries]
        self.memory.state["tool_handle_count"] = len(self.memory.entries)
        is_error_payload = isinstance(output, dict) and output.get("ok") is False
        content = (
            json.dumps(output)
            if is_error_payload
            else json.dumps({"handle": entry.handle, "summary": entry.summary})
        )
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id or tool_name,
            "content": content,
        }
        self._append_message(tool_message)
        if self.trace:
            if arguments is not None:
                self.trace.record_tool_call(tool_name, arguments)
            self.trace.record_tool_result(tool_name, entry.handle, entry.summary)
        return entry

    def _run_tool(self, tool_name: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
        tool = self.registry.get(tool_name)
        if tool is None:
            return {"ok": False, "error": "tool not found"}, False
        if not self.policy.is_tool_allowed(tool_name):
            return {"ok": False, "error": "tool not allowed"}, False
        return self._execute_tool(tool, arguments)

    def _record_tool_error(self, state: AgentState, tool_name: str) -> None:
        counts = state.memory_state.get("tool_error_counts", {})
        counts[tool_name] = counts.get(tool_name, 0) + 1
        state.memory_state["tool_error_counts"] = counts
        penalties = state.memory_state.get("route_penalties", {})
        penalties[tool_name] = penalties.get(tool_name, 0) + 1
        state.memory_state["route_penalties"] = penalties
        self._increment_tool_failure(state, tool_name)
        state.last_error = {
            "error": f"Tool error from {tool_name}",
            "suggested_fix": "Adjust arguments or try an alternative tool.",
        }
        self._record_failure(
            FailureEvent(
                tag=FailureTag.TOOL_ERROR.value,
                reason=f"Tool error from {tool_name}",
                details={"tool_name": tool_name},
            )
        )
        if state.memory_state.get("last_routed_tool") == tool_name:
            self._record_failure(
                FailureEvent(
                    tag=FailureTag.ROUTER_MISFIRE.value,
                    reason="Router selected a tool that failed",
                    details={"tool_name": tool_name},
                )
            )

    def _record_failure(self, failure: FailureEvent) -> None:
        if not self.trace:
            return
        payload = {"tag": failure.tag, "reason": failure.reason}
        if failure.details:
            payload["details"] = failure.details
        self.trace.record("failure", payload)

    def _record_verifier_result(
        self, state: AgentState, task: MicroTask, result: "VerifierResult"
    ) -> None:
        failures = state.memory_state.get("verifier_failures", {})
        if not result.passed:
            failures[task.id] = failures.get(task.id, 0) + 1
            failure_payload = [
                {
                    "check_name": failure.check_name,
                    "reason": failure.reason,
                    "expected": failure.expected,
                    "got": failure.got,
                    "minimal_fix": failure.minimal_fix,
                }
                for failure in result.failures
            ]
            state.last_error = {
                "error": "Verification failed",
                "failures": failure_payload,
            }
            state.memory_state["needs_revision"] = True
            constraints = state.memory_state.setdefault("constraints", {})
            if isinstance(constraints, dict):
                constraints["verifier_failures"] = failure_payload
        if not result.passed and state.memory_state.get("candidate_source") == "tool":
            tool_name = state.memory_state.get("last_tool_name") or "tool"
            penalties = state.memory_state.get("route_penalties", {})
            penalties[tool_name] = penalties.get(tool_name, 0) + 1
            state.memory_state["route_penalties"] = penalties
            self._increment_tool_failure(state, tool_name)
            self._record_failure(
                FailureEvent(
                    tag=FailureTag.ROUTER_MISFIRE.value,
                    reason="Tool output failed verification",
                    details={"tool_name": tool_name},
                )
            )
            for failure in result.failures:
                self._record_failure(
                    FailureEvent(
                        tag=verifier_failure_tag(failure.check_name),
                        reason=failure.reason,
                    )
                )
        if result.passed:
            state.last_error = None
            state.memory_state["needs_revision"] = False
            constraints = state.memory_state.get("constraints")
            if isinstance(constraints, dict):
                constraints.pop("verifier_failures", None)
        state.memory_state["verifier_failures"] = failures
        if self.trace:
            self.trace.record(
                "verifier",
                {
                    "task_id": task.id,
                    "ok": result.passed,
                    "failures": [
                        {
                            "check_name": failure.check_name,
                            "reason": failure.reason,
                            "expected": failure.expected,
                            "got": failure.got,
                            "minimal_fix": failure.minimal_fix,
                        }
                        for failure in result.failures
                    ],
                    "checks_run": result.checks_run,
                },
            )
            if state.last_error and state.last_error.get("failures"):
                self.trace.record(
                    "verifier_issues",
                    {
                        "task_id": task.id,
                        "issues": state.last_error["failures"][:3],
                    },
                )
        state.task_history.append(state.task_graph.model_copy(deep=True))

    def _increment_tool_failure(self, state: AgentState, tool_name: str) -> None:
        counts = state.memory_state.get("tool_failure_counts", {})
        counts[tool_name] = counts.get(tool_name, 0) + 1
        state.memory_state["tool_failure_counts"] = counts
        if counts[tool_name] >= 2:
            # Hard-disable tools after repeated failures to avoid routing loops.
            disabled = set(state.memory_state.get("disabled_tools", []))
            if tool_name not in disabled:
                disabled.add(tool_name)
                state.memory_state["disabled_tools"] = sorted(disabled)
                if self.trace:
                    self.trace.record(
                        "tool_disabled",
                        {"tool_name": tool_name, "reason": "failure_count"},
                    )

    def _record_tool_execution(
        self,
        state: AgentState,
        current_task: MicroTask | None,
        tool_name: str,
    ) -> None:
        event_id = state.memory_state.get("tool_event_id", 0) + 1
        state.memory_state["tool_event_id"] = event_id
        state.memory_state["last_executed_tool"] = tool_name
        state.memory_state["last_executed_task_id"] = current_task.id if current_task else None
        state.memory_state["last_task_tool_hint"] = (
            current_task.tool_hint if current_task else None
        )

    def _record_verification_event(
        self,
        state: AgentState,
        task: MicroTask,
        ok: bool,
    ) -> None:
        event_id = state.memory_state.get("verification_event_id", 0) + 1
        state.memory_state["verification_event_id"] = event_id
        state.memory_state["last_verification_task_id"] = task.id
        state.memory_state["last_verification_ok"] = ok

    def _is_forced_tool_task(self, task: MicroTask) -> bool:
        tool_hint = (task.tool_hint or "").strip()
        if not tool_hint or tool_hint.lower() == "router":
            return False
        if self._check_includes_type(task.check, "code_run"):
            return False
        return self.registry.get(tool_hint) is not None

    def _update_progress(self, state: AgentState, result: "VerifierResult") -> None:
        facts = state.memory_state.get("facts")
        fact_count = len(facts) if isinstance(facts, list) else 0
        fact_signature = "|".join(facts) if isinstance(facts, list) else ""
        structured_facts = state.memory_state.get("facts_structured")
        structured_count = len(structured_facts) if isinstance(structured_facts, list) else 0
        constraints = state.memory_state.get("constraints")
        constraints_signature = (
            json.dumps(constraints, sort_keys=True)
            if isinstance(constraints, dict)
            else None
        )
        intermediates = state.memory_state.get("intermediates")
        intermediates_signature = (
            "|".join(sorted(intermediates.keys()))
            if isinstance(intermediates, dict)
            else None
        )
        tool_count = len(self.memory.entries)
        issue_count = len(result.failures) if result else None
        state.progress.update(
            fact_count,
            tool_count,
            result.passed,
            issue_count,
            fact_signature=fact_signature,
            structured_fact_count=structured_count,
            constraints_signature=constraints_signature,
            intermediates_signature=intermediates_signature,
        )

    def _extract_constraints_from_query(self, query: str) -> dict[str, Any]:
        requirements: list[str] = []
        prohibitions: list[str] = []
        sentences = re.split(r"[.!?]\s+", query)
        for sentence in sentences:
            text = sentence.strip()
            if not text:
                continue
            lowered = text.lower()
            if any(token in lowered for token in ["must", "should", "need to", "required"]):
                requirements.append(text)
            if any(token in lowered for token in ["do not", "don't", "avoid", "never"]):
                prohibitions.append(text)
        constraints: dict[str, Any] = {}
        if requirements:
            constraints["requirements"] = requirements[:8]
        if prohibitions:
            constraints["prohibitions"] = prohibitions[:8]
        return constraints

    def _apply_backtrack(self, state: AgentState, reason: str) -> None:
        state.budgets.backtracks = max(0, state.budgets.backtracks - 1)
        state.memory_state["backtrack_count"] = state.memory_state.get("backtrack_count", 0) + 1
        snapshot = state.snapshots.pop() if state.snapshots else None
        if snapshot:
            state.task_graph = snapshot.task_graph.model_copy(deep=True)
            for key, value in snapshot.memory_state_subset.items():
                state.memory_state[key] = value
            state.memory_state["tool_error_counts"] = snapshot.tool_error_counts
            state.memory_state["verifier_failures"] = snapshot.verifier_failures
            state.last_tool_summary = snapshot.last_tool_summary
            state.last_error = snapshot.last_error
            state.progress = ProgressTracker.from_dict(snapshot.progress_state)
            handle_count = snapshot.tool_handle_count
            if handle_count:
                self.memory.entries = self.memory.entries[:handle_count]
                handles = set(snapshot.tool_handles)
                if handles:
                    self.memory.raw_outputs = {
                        key: value
                        for key, value in self.memory.raw_outputs.items()
                        if key in handles
                    }
            else:
                self.memory.entries = []
                self.memory.raw_outputs = {}
            if self.trace:
                self.trace.record(
                    "snapshot_restore",
                    {"task_id": state.task_graph.current_task_id, "handles": handle_count},
                )
            state.last_snapshot_task_id = state.task_graph.current_task_id
        elif state.task_history:
            state.task_graph = state.task_history.pop()
        else:
            current_task = state.task_graph.current_task or state.task_graph.next_task()
            if current_task:
                state.task_graph.mark_failed(current_task.id, notes="Backtracked")
        state.last_error = {
            "error": "Backtracked",
            "suggested_fix": "Try a different approach or tool.",
        }
        state.memory_state["pending_verification"] = False
        state.progress.reset()
        if self.trace:
            self.trace.record(
                "backtrack",
                {"reason": reason, "task_id": state.task_graph.current_task_id},
            )

    def _attach_code_source(self, task: MicroTask, source: Any) -> MicroTask:
        def attach(check: CheckSpec) -> CheckSpec:
            if check.type == "code_run":
                params = dict(check.params)
                params["source"] = source or ""
                return check.model_copy(update={"params": params})
            if check.all_of:
                return check.model_copy(update={"all_of": [attach(sub) for sub in check.all_of]})
            if check.any_of:
                return check.model_copy(update={"any_of": [attach(sub) for sub in check.any_of]})
            return check

        updated = task.model_copy(deep=True)
        updated.check = attach(updated.check)
        return updated

    def _attach_tool_recompute(self, task: MicroTask, state: AgentState) -> MicroTask:
        tool_name = state.memory_state.get("last_tool_name")
        tool_args = state.memory_state.get("last_tool_args")

        def attach(check: CheckSpec) -> CheckSpec:
            if check.type == "tool_recompute":
                params = dict(check.params)
                if tool_name and isinstance(tool_args, dict):
                    params.setdefault("tool_name", tool_name)
                    params.setdefault("tool_args", tool_args)
                return check.model_copy(update={"params": params})
            if check.all_of:
                return check.model_copy(update={"all_of": [attach(sub) for sub in check.all_of]})
            if check.any_of:
                return check.model_copy(update={"any_of": [attach(sub) for sub in check.any_of]})
            return check

        updated = task.model_copy(deep=True)
        updated.check = attach(updated.check)
        return updated

    def _attach_tool_error_absent(self, task: MicroTask, state: AgentState) -> MicroTask:
        if not self._check_includes_type(task.check, "tool_error_absent"):
            return task

        def attach(check: "CheckSpec") -> "CheckSpec":
            if check.type == "tool_error_absent":
                params = dict(check.params)
                params["tool_output"] = state.memory_state.get("last_tool_output")
                return check.model_copy(update={"params": params})
            if check.all_of:
                return check.model_copy(
                    update={"all_of": [attach(sub) for sub in check.all_of]}
                )
            if check.any_of:
                return check.model_copy(
                    update={"any_of": [attach(sub) for sub in check.any_of]}
                )
            return check

        updated = task.model_copy(deep=True)
        updated.check = attach(updated.check)
        return updated

    def _direct_tool_args(
        self,
        tool_name: str,
        query: str,
        last_parse_failed: str | None,
    ) -> dict[str, Any] | None:
        if tool_name == "calculator":
            cue_match = re.search(r"(?:calc\s*:|=)\s*(.+)$", query, re.IGNORECASE)
            if cue_match:
                expression = cue_match.group(1).strip()
                if expression:
                    return {"expression": expression}
            matches = re.findall(r"[0-9\.\s\+\-\*\/\(\)]+", query)
            if matches:
                expression = max(matches, key=len).strip()
                if expression:
                    return {"expression": expression}
            return {"expression": query.strip()}
        if tool_name == "json_repair":
            if last_parse_failed:
                return {"text": last_parse_failed}
            if re.search(r"\bfix json\b", query, re.IGNORECASE):
                return {"text": query}
            return None
        if tool_name == "http_fetch":
            match = re.search(r"https?://\S+", query)
            if match:
                return {"url": match.group(0)}
        if tool_name == "regex_extract":
            pattern_match = re.search(r"/(.+?)/", query)
            if pattern_match:
                pattern = pattern_match.group(1)
            else:
                pattern_match = re.search(
                    r"(?:pattern|regex)\s*:\s*(.+)$", query, re.IGNORECASE
                )
                pattern = pattern_match.group(1).strip() if pattern_match else r".+"
            quoted = re.findall(r"(['\"])(.+?)\1", query, re.DOTALL)
            if quoted:
                text = max((segment for _, segment in quoted), key=len)
            elif self.memory.state.get("last_tool_summary"):
                text = str(self.memory.state["last_tool_summary"])
            else:
                text = query
            return {"pattern": pattern, "text": text}
        if tool_name == "unit_convert":
            match = re.search(
                r"convert\s+([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)\s+to\s+([a-zA-Z]+)",
                query,
                re.IGNORECASE,
            )
            if match:
                return {
                    "value": float(match.group(1)),
                    "from_unit": match.group(2),
                    "to_unit": match.group(3),
                }
        return None

    def _run_model_candidates(
        self,
        call_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        state: AgentState,
        current_task: MicroTask | None,
        verifier: Verifier,
        strict_json_mode: bool,
    ):
        if (
            self.branch_candidates <= 1
            or current_task is None
            or not self._is_objective_check(current_task.check)
            or current_task.tool_hint
        ):
            return self._call_model(call_messages, tools, state)
        available = min(state.budgets.model_calls, self.max_model_calls - self._model_calls)
        effective_k = max(1, min(self.branch_candidates, available, 3))
        if effective_k <= 1:
            return self._call_model(call_messages, tools, state)
        candidates: list[dict[str, Any]] = []
        responses = []
        for idx in range(effective_k):
            label = chr(ord("A") + idx)
            response = self._call_model(
                call_messages,
                tools,
                state,
                extra_system_message=f"Draft {label}",
            )
            ok, issues = self._evaluate_candidate_response(
                response,
                current_task,
                verifier,
                strict_json_mode,
                state,
            )
            candidates.append({"label": label, "ok": ok, "issues": issues})
            responses.append(response)
        selected_index = self._select_candidate_index(candidates)
        if self.trace:
            self.trace.record(
                "branch_select",
                {
                    "task_id": current_task.id,
                    "selected": candidates[selected_index]["label"],
                    "candidates": [
                        {
                            "label": candidate["label"],
                            "ok": candidate["ok"],
                            "issues": candidate["issues"][:3],
                        }
                        for candidate in candidates
                    ],
                },
            )
        return responses[selected_index]

    def _call_model(
        self,
        call_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        state: AgentState,
        extra_system_message: str | None = None,
    ):
        messages = list(call_messages)
        if extra_system_message:
            messages.append({"role": "system", "content": extra_system_message})
        if self._model_calls >= self.max_model_calls or state.budgets.model_calls <= 0:
            return ModelResponse(final_text=None)
        response = self.model.chat(messages, tools=tools)
        self._model_calls += 1
        state.budgets.model_calls -= 1
        if self.trace:
            tool_payload = (
                response.tool_call.model_dump()
                if response.tool_call is not None
                else None
            )
            self.trace.record_model_response(response.final_text, tool_payload)
        return response

    def _select_candidate_index(self, candidates: list[dict[str, Any]]) -> int:
        scored = []
        for idx, candidate in enumerate(candidates):
            ok = candidate["ok"]
            issues = candidate["issues"] or []
            scored.append((0 if ok else 1, len(issues), idx))
        scored.sort()
        return scored[0][2]

    def _evaluate_candidate_response(
        self,
        response,
        task: MicroTask,
        verifier: Verifier,
        strict_json_mode: bool,
        state: AgentState,
    ) -> tuple[bool, list[str]]:
        if response.tool_call is not None and response.final_text is None:
            return False, ["Model returned a tool call"]
        if response.final_text is None:
            return False, ["No candidate output"]
        protocol = self._parse_model_protocol(response.final_text)
        if protocol is None and strict_json_mode:
            return False, ["Output is not valid JSON"]
        if isinstance(protocol, ProtocolFinal):
            candidate_output = format_final(
                protocol.answer, protocol.checks, eval_mode=self.eval_mode
            )
        else:
            candidate_output = response.final_text
        candidate_for_check = self._coerce_candidate_for_check(task, candidate_output)
        check_task = task
        if self._check_includes_type(check_task.check, "code_run"):
            check_task = self._attach_code_source(check_task, candidate_output)
        check_task = self._attach_tool_error_absent(check_task, state)
        if self._check_includes_type(check_task.check, "tool_recompute"):
            check_task = self._attach_tool_recompute(check_task, state)
        result = verifier.verify(candidate_for_check, check_task)
        return result.passed, [failure.reason for failure in result.failures]

    def _is_objective_check(self, check: CheckSpec) -> bool:
        objective_predicates = {
            "non_empty",
            "looks_numeric",
            "numeric_range",
            "contains_fields",
            "unit_present",
            "mcq_choice",
        }
        if check.type in {
            "schema",
            "contains_fields",
            "regex",
            "tool_recompute",
            "code_run",
            "exact",
            "numeric_tolerance",
            "unit_sanity",
        }:
            return True
        if check.type == "predicate":
            return check.params.get("name") in objective_predicates
        if check.all_of:
            return any(self._is_objective_check(sub) for sub in check.all_of)
        if check.any_of:
            return any(self._is_objective_check(sub) for sub in check.any_of)
        return False

    def _build_contract_message(
        self,
        state: AgentState,
        hint: str | None,
        retry_instruction: dict[str, Any] | None,
        format_retry_message: dict[str, Any] | None,
        strict_json_mode: bool,
    ) -> dict[str, Any] | None:
        current_task = state.task_graph.current_task
        if current_task is None:
            return None
        contract_payload = self._build_contract_payload(current_task, state)
        parts: list[str] = []
        parts.append("Microtask contract:")
        parts.append(f"- Goal: {current_task.goal}")
        criteria = self._describe_check(current_task.check)
        if criteria:
            parts.append(f"- Success: {criteria}")
        if current_task.tool_hint:
            parts.append(f"- Tool hint: {current_task.tool_hint}")
        if state.last_error and state.last_error.get("failures"):
            failures = state.last_error.get("failures") or []
            constraint_lines = []
            for failure in failures:
                reason = failure.get("reason")
                minimal_fix = failure.get("minimal_fix") or "Fix the failing check."
                check_name = failure.get("check_name") or "check"
                constraint_lines.append(
                    f"{check_name}: {reason} | Fix: {minimal_fix}"
                )
            if constraint_lines:
                parts.append("- Failing checks:")
                parts.extend(f"  - {line}" for line in constraint_lines)
                parts.append(
                    "- Hard constraints: only satisfy the failing checks above."
                )
        if strict_json_mode:
            parts.append("- Output: single JSON object only")
        if hint:
            parts.append(f"- Hint: {hint}")
        if retry_instruction and retry_instruction.get("content"):
            parts.append(f"- Tool retry: {retry_instruction['content']}")
        if format_retry_message and format_retry_message.get("content"):
            parts.append(f"- Format retry: {format_retry_message['content']}")
        parts.append("CONTRACT_JSON:")
        parts.append(json.dumps(contract_payload, ensure_ascii=False))
        content = "\n".join(parts)
        if len(content) > 1200:
            content = self._trim_contract_content(content, contract_payload, parts)
        return {"role": "user", "content": content}

    def _build_contract_payload(
        self,
        task: MicroTask,
        state: AgentState,
    ) -> dict[str, Any]:
        required_fields = self._extract_required_fields(task.check)
        expected_type = "tool" if task.tool_hint else "final"
        payload = {
            "task_id": task.id,
            "goal": self._truncate_text(task.goal, 240),
            "expected_output": {
                "type": expected_type,
                "required_fields": required_fields,
            },
            "check": self._truncate_check(task.check),
            "tool_hint": self._truncate_text(task.tool_hint or "", 160),
            "stop_when": "verifier_ok",
        }
        if state.last_error and state.last_error.get("failures"):
            payload["verifier_failures"] = state.last_error.get("failures")[:3]
        return payload

    def _extract_required_fields(self, check: CheckSpec) -> list[str]:
        fields: list[str] = []
        if check.type == "contains_fields":
            fields.extend(check.params.get("required_fields") or [])
        elif check.type == "json_protocol":
            fields.extend(check.params.get("required_keys") or ["answer"])
        elif check.type == "schema":
            schema = check.params.get("schema") or {}
            if isinstance(schema, dict):
                required = schema.get("required") or []
                fields.extend(required)
        if check.all_of:
            for sub in check.all_of:
                fields.extend(self._extract_required_fields(sub))
        if check.any_of:
            for sub in check.any_of:
                fields.extend(self._extract_required_fields(sub))
        return list(dict.fromkeys(fields))

    def _truncate_check(self, check: CheckSpec) -> dict[str, Any]:
        payload = check.model_dump()
        return self._truncate_payload(payload, max_str_len=160, max_list_items=4, max_depth=3)

    def _truncate_payload(
        self,
        payload: Any,
        max_str_len: int,
        max_list_items: int,
        max_depth: int,
    ) -> Any:
        if max_depth <= 0:
            return payload
        if isinstance(payload, dict):
            return {
                key: self._truncate_payload(value, max_str_len, max_list_items, max_depth - 1)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            trimmed = payload[:max_list_items]
            return [
                self._truncate_payload(item, max_str_len, max_list_items, max_depth - 1)
                for item in trimmed
            ]
        if isinstance(payload, str):
            return self._truncate_text(payload, max_str_len)
        return payload

    def _truncate_text(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _trim_contract_content(
        self,
        content: str,
        contract_payload: dict[str, Any],
        parts: list[str],
    ) -> str:
        payload = dict(contract_payload)
        payload["goal"] = self._truncate_text(str(payload.get("goal") or ""), 120)
        payload["tool_hint"] = self._truncate_text(str(payload.get("tool_hint") or ""), 80)
        payload["check"] = {"type": payload.get("check", {}).get("type")}
        parts = parts[:-1] + [json.dumps(payload, ensure_ascii=False)]
        trimmed = "\n".join(parts)
        if len(trimmed) > 1200:
            retry_parts = [
                part
                for part in parts
                if part.startswith("- Tool retry:") or part.startswith("- Format retry:")
            ]
            minimal_parts = [
                "Microtask contract:",
                *retry_parts,
                "CONTRACT_JSON:",
                json.dumps(payload, ensure_ascii=False),
            ]
            trimmed = "\n".join(minimal_parts)
        return trimmed

    def _describe_check(self, check: "CheckSpec") -> str:
        if check.all_of:
            return " AND ".join(self._describe_check(sub) for sub in check.all_of)
        if check.any_of:
            return " OR ".join(self._describe_check(sub) for sub in check.any_of)
        if check.type == "contains_fields":
            fields = check.params.get("required_fields") or []
            return f"include fields: {', '.join(fields) or 'unspecified'}"
        if check.type == "json_protocol":
            keys = check.params.get("required_keys") or ["answer"]
            return f"valid JSON with keys: {', '.join(keys)}"
        if check.type == "regex":
            pattern = check.params.get("pattern") or ""
            return f"match /{pattern}/" if pattern else "match required pattern"
        if check.type == "predicate":
            name = check.params.get("name") or "non_empty"
            return f"predicate: {name}"
        if check.type == "tool_recompute":
            return "match tool recompute output"
        if check.type == "tool_error_absent":
            return "no tool errors"
        if check.type == "code_run":
            return "code runs without errors"
        if check.type == "schema":
            return "match required schema"
        return "satisfy check"

    def _check_includes_type(self, check: "CheckSpec", check_type: str) -> bool:
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

    def _coerce_candidate_for_check(self, task: MicroTask, candidate: Any) -> Any:
        if candidate is None:
            return candidate
        if not isinstance(candidate, dict):
            required_fields = self._extract_required_fields(task.check)
            if "answer" in required_fields:
                return {"answer": str(candidate)}
        return candidate

    def _update_facts_from_tool(
        self, tool_name: str, output: Any, summary_text: str
    ) -> None:
        handle = self.memory.entries[-1].handle if self.memory.entries else None
        facts = extract_facts(tool_name, output, summary_text, source=handle)
        structured = extract_facts_structured(
            tool_name, output, summary_text, source=handle
        )
        if not facts:
            self.memory.state["last_tool_facts"] = []
        else:
            fact_store = self.memory.state.setdefault("facts", [])
            if not isinstance(fact_store, list):
                fact_store = []
                self.memory.state["facts"] = fact_store
            existing = set(fact_store)
            for fact in facts:
                if fact not in existing:
                    fact_store.append(fact)
                    existing.add(fact)
            self.memory.state["last_tool_facts"] = facts
        if structured:
            structured_store = self.memory.state.setdefault("facts_structured", [])
            if not isinstance(structured_store, list):
                structured_store = []
                self.memory.state["facts_structured"] = structured_store
            existing_structured = {
                (item.get("kind"), item.get("value")) for item in structured_store
            }
            for item in structured:
                key = (item.get("kind"), item.get("value"))
                if key in existing_structured:
                    continue
                structured_store.append(item)
                existing_structured.add(key)
            self.memory.state["last_tool_facts_structured"] = structured
        self._update_intermediates_from_facts(structured)

    def _update_intermediates_from_facts(
        self, structured: list[dict[str, Any]]
    ) -> None:
        intermediates = self.memory.state.setdefault("intermediates", {})
        if not isinstance(intermediates, dict):
            intermediates = {}
            self.memory.state["intermediates"] = intermediates
        index = 0
        for fact in structured:
            kind = fact.get("kind")
            value = fact.get("value")
            if not isinstance(value, str):
                continue
            if kind == "kv" and ":" in value:
                key, val = value.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key and key not in intermediates:
                    intermediates[key] = val
            elif kind == "number":
                key = f"number_{index}"
                if key not in intermediates:
                    intermediates[key] = value
                    index += 1

    def _pick_best(self, candidates: list[AgentResult]) -> AgentResult:
        def score(candidate: AgentResult) -> tuple[float, int, int]:
            confidence = candidate.confidence or 0.0
            check_len = len(candidate.checks)
            answer_len = len(candidate.answer)
            return (confidence, check_len, answer_len)

        return max(candidates, key=score)

    def _verify_answer(self, query: str, answer: str) -> AgentResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "Verify the answer. Respond with JSON only using "
                    '{"type":"final","answer":"...","scratchpad":"...","confidence":0.0,"checks":[...]} '
                    "No chain-of-thought."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\nAnswer: {answer}",
            },
        ]
        response = self.model.chat(messages, tools=None)
        self._model_calls += 1
        if response.final_text:
            protocol = parse_protocol(response.final_text)
            if isinstance(protocol, ProtocolFinal):
                if protocol.scratchpad and self.trace:
                    self.trace.record("scratchpad", {"content": protocol.scratchpad})
                return AgentResult(
                    answer=format_final(
                        protocol.answer, protocol.checks, eval_mode=self.eval_mode
                    ),
                    checks=protocol.checks,
                    confidence=protocol.confidence,
                )
        return AgentResult(answer=answer)

    def _finalize_trace(self) -> str | None:
        if not self.trace:
            return None
        stats = {
            "model_calls": self._model_calls,
            "tool_calls": len(self.tools_used),
        }
        return self.trace.finalize(stats)

    def _execute_tool(self, tool: Tool, arguments: dict[str, Any]) -> tuple[Any, bool]:
        try:
            result = tool.run(arguments)
            output = result.output
            ok = not (isinstance(output, dict) and output.get("ok") is False)
            return output, ok
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            return self._tool_error_payload(tool, exc), False
        except Exception as exc:  # noqa: BLE001
            return self._tool_error_payload(tool, exc), False

    def _tool_error_payload(self, tool: Tool, exc: Exception) -> dict[str, Any]:
        schema = None
        if getattr(tool, "input_schema", None) is not None:
            schema = tool.input_schema.model_json_schema()
        return {
            "ok": False,
            "tool": tool.name,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "expected_input_schema": schema,
            "hint": "Fix arguments and retry",
        }

    def _build_retry_instruction(
        self, tool: Tool, output: Any
    ) -> dict[str, Any]:
        error_type = "ToolError"
        error_message = "Unknown error"
        if isinstance(output, dict):
            error_type = str(output.get("error_type") or error_type)
            error_message = str(output.get("error") or error_message)
        error_message = error_message.strip()
        if len(error_message) > 160:
            error_message = f"{error_message[:157]}..."
        schema = None
        if getattr(tool, "input_schema", None) is not None:
            schema = tool.input_schema.model_json_schema()
        required_fields: list[str] = []
        properties: dict[str, Any] = {}
        if isinstance(schema, dict):
            required_fields = list(schema.get("required") or [])
            properties = schema.get("properties") or {}
        example_args = {
            field: self._example_value(properties.get(field, {}))
            for field in required_fields
        }
        required_display = ", ".join(required_fields) if required_fields else "none"
        example_call = {
            "type": "tool",
            "name": tool.name,
            "arguments": example_args,
        }
        content = (
            f"Tool error for '{tool.name}'.\n"
            f"Error: {error_type}: {error_message}\n"
            f"Required fields: {required_display}\n"
            f"Example tool call JSON: {json.dumps(example_call)}\n"
            "Return ONLY a tool call JSON object with corrected arguments."
        )
        return {"role": "user", "content": content}

    def _build_format_retry_message(self) -> dict[str, Any]:
        tool_example = {"type": "tool", "name": "tool_name", "arguments": {}}
        final_example = {
            "type": "final",
            "answer": "ok",
            "confidence": 0.5,
            "checks": [],
        }
        content = (
            "Format error: previous response was not valid JSON. "
            "Return ONLY a single JSON object matching the protocol.\n"
            f"Tool example: {json.dumps(tool_example)}\n"
            f"Final example: {json.dumps(final_example)}"
        )
        return {"role": "system", "content": content}

    def _example_value(self, schema: dict[str, Any]) -> Any:
        schema_type = schema.get("type")
        if schema_type == "string":
            return "<value>"
        if schema_type == "integer":
            return 0
        if schema_type == "number":
            return 0.0
        if schema_type == "boolean":
            return False
        if schema_type == "array":
            return []
        if schema_type == "object":
            return {}
        return "<value>"

    def _parse_model_protocol(
        self, content: str
    ) -> ProtocolToolCall | ProtocolFinal | None:
        if not self.strict_json_mode:
            return parse_protocol(content)
        stripped = content.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                payload = repair_json(content)
            except JsonRepairError:
                return None
        if not isinstance(payload, dict):
            return None
        return protocol_from_payload(payload)

    def _code_check_loop(self, answer: str) -> str:
        current_answer = answer
        for attempt in range(self.code_check_max_iters):
            code_blocks = self._extract_python_blocks(current_answer)
            if not code_blocks:
                return current_answer
            run_result = self._run_code_blocks(code_blocks)
            if run_result["ok"]:
                return current_answer
            if attempt + 1 >= self.code_check_max_iters:
                return current_answer
            error_text = run_result["error"] or "Unknown error"
            self._append_message(
                {
                    "role": "user",
                    "content": (
                        "The Python code failed to run:\n"
                        f"{error_text}\n"
                        "Please return a corrected solution with updated code. "
                        "Respond with JSON only and no additional commentary."
                    ),
                }
            )
            if self._model_calls >= self.max_model_calls:
                return current_answer
            response = self.model.chat(self._messages, tools=None)
            self._model_calls += 1
            if self.trace:
                tool_payload = (
                    response.tool_call.model_dump()
                    if response.tool_call is not None
                    else None
                )
                self.trace.record_model_response(response.final_text, tool_payload)
            if response.final_text:
                protocol = self._parse_model_protocol(response.final_text)
                if protocol is None and self.strict_json_mode:
                    if self._extract_python_blocks(response.final_text):
                        current_answer = response.final_text
                        continue
                    return current_answer
                if isinstance(protocol, ProtocolFinal):
                    current_answer = format_final(
                        protocol.answer, protocol.checks, eval_mode=self.eval_mode
                    )
                elif response.final_text:
                    current_answer = response.final_text
        return current_answer

    def _extract_python_blocks(self, text: str) -> list[str]:
        code_blocks: list[str] = []
        for match in re.finditer(
            r"```(?P<lang>[a-zA-Z0-9_-]*)\s*\n(?P<code>.*?)```",
            text,
            re.DOTALL,
        ):
            lang = (match.group("lang") or "").strip().lower()
            if lang and not lang.startswith("python"):
                continue
            code_blocks.append(match.group("code").strip())
        return code_blocks

    def _run_code_blocks(self, code_blocks: list[str]) -> dict[str, Any]:
        if len(code_blocks) > 1:
            files: dict[str, str] = {}
            for idx, block in enumerate(code_blocks):
                lines = block.splitlines()
                file_path = None
                if lines:
                    match = re.match(r"#\s*file(name)?:\s*(\S+)", lines[0].strip(), re.IGNORECASE)
                    if match:
                        file_path = match.group(2)
                        lines = lines[1:]
                if not file_path:
                    file_path = f"snippet_{idx}.py"
                files[file_path] = "\n".join(lines).strip() + "\n"
            main_file = "main.py" if "main.py" in files else next(iter(files))
            tool = self.registry.get("code_run_multi")
            if tool is None:
                return {"ok": False, "error": "code_run_multi tool not available"}
            result = tool.run(
                {
                    "files": files,
                    "command": f"{sys.executable} {main_file}",
                    "timeout_seconds": 10,
                }
            )
            output = result.output
            ok = output.get("exit_code") == 0
            error = output.get("stderr") or output.get("stdout")
            return {"ok": ok, "error": error}
        tool = self.registry.get("python_sandbox")
        if tool is None:
            return {"ok": False, "error": "python_sandbox tool not available"}
        result = tool.run({"code": code_blocks[0], "timeout_seconds": 2})
        output = result.output
        error = output.get("error")
        return {"ok": error is None, "error": error}


def protocol_to_toolcall(protocol: ProtocolToolCall) -> "ToolCall":
    from agentforge.models.base import ToolCall

    return ToolCall(name=protocol.name, arguments=protocol.arguments)
