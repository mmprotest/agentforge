"""Core agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import re
import sys
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from agentforge.models.base import BaseChatModel
from agentforge.protocol import (
    ProtocolFinal,
    ProtocolToolCall,
    format_final,
    parse_protocol,
    protocol_from_payload,
)
from agentforge.epistemic import EpistemicSource, Fact
from agentforge.memory import MemoryStore
from agentforge.problem_state import ProblemState
from agentforge.routing import (
    is_code_task,
    is_route_ambiguous,
    should_enable_tools,
    suggest_tool,
    tool_candidates,
)
from agentforge.safety.policy import SafetyPolicy
from agentforge.task_types import EPISTEMIC_REQUIREMENTS, TaskType, determine_task_type
from agentforge.trace import TraceRecorder
from agentforge.tools.base import Tool
from agentforge.tools.builtins.deep_think import DeepThinkTool
from agentforge.tools.registry import ToolRegistry
from agentforge.util.context_trim import trim_messages
from agentforge.util.json_repair import JsonRepairError, repair_json
from agentforge.util.logging import get_logger, redact


logger = get_logger(__name__)
@dataclass
class AgentResult:
    answer: str
    tools_used: list[str] = field(default_factory=list)
    tools_created: list[str] = field(default_factory=list)
    trace_path: str | None = None
    checks: list[str] = field(default_factory=list)
    confidence: float | None = None


@dataclass
class StepOutcome:
    step_summary: str
    candidate_solution: str | None = None
    failure_reason: str | None = None
    fact: Fact | None = None


class Agent:
    """Agent that can call tools and optionally create new tools."""

    def __init__(
        self,
        model: BaseChatModel,
        registry: ToolRegistry,
        policy: SafetyPolicy | None = None,
        mode: str = "direct",
        verify: bool = False,
        self_consistency: int = 1,
        max_model_calls: int | None = None,
        max_steps: int | None = None,
        max_tool_calls: int | None = None,
        memory: MemoryStore | None = None,
        trace: TraceRecorder | None = None,
        strict_json_mode: bool = False,
        max_message_chars: int = 24000,
        max_turns: int = 20,
        trim_strategy: str = "drop_oldest",
        code_check: bool = False,
        code_check_max_iters: int = 2,
    ) -> None:
        self.model = model
        self.registry = registry
        self.policy = policy or SafetyPolicy()
        self.mode = mode
        self.verify = verify
        self.self_consistency = max(1, min(self_consistency, 3))
        self.max_model_calls = max_model_calls or self.policy.max_model_calls
        self.max_steps = max_steps or self.policy.max_steps
        self.max_tool_calls = max_tool_calls or self.policy.max_tool_calls
        self.memory = memory or MemoryStore()
        self.trace = trace
        self.strict_json_mode = strict_json_mode
        self.max_message_chars = max_message_chars
        self.max_turns = max_turns
        self.trim_strategy = trim_strategy
        self.code_check = code_check
        self.code_check_max_iters = max(1, code_check_max_iters)
        self.tools_used: list[str] = []
        self.tools_created: list[str] = []
        self._model_calls = 0
        self._tool_calls = 0
        self._pending_tool_call_response = None
        self._pending_model_response = None
        self._mcq_letters: set[str] = set()

    def _internal_plan(self, query: str) -> None:
        if self.mode != "deep":
            return
        deep_tool = self.registry.get("deep_think")
        if isinstance(deep_tool, DeepThinkTool):
            result = deep_tool.run({"problem": query, "constraints": []})
            plan = result.output
        else:
            plan = {"plan": ["Use available tools"], "checks": []}
        self._append_message(
            {"role": "system", "content": f"Internal plan (do not reveal): {plan}"}
        )

    def _append_message(self, message: dict[str, Any]) -> None:
        self._messages.append(message)
        if self.trim_strategy == "drop_oldest":
            self._messages = trim_messages(
                self._messages, self.max_message_chars, self.max_turns
            )

    def run(self, query: str) -> AgentResult:
        logger.info("Agent run started (mode=%s, verify=%s).", self.mode, self.verify)
        self._model_calls = 0
        self._tool_calls = 0
        self._pending_tool_call_response = None
        self._mcq_letters = self._detect_mcq(query)
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
            if self._mcq_letters:
                best = self._pick_majority_mcq(candidates)
            else:
                best = self._pick_best(candidates)
            if self.verify and self._model_calls < self.max_model_calls:
                verified = self._verify_answer(query, best.answer)
                best.answer = verified.answer
                best.checks = verified.checks
                best.confidence = verified.confidence
            if self._mcq_letters:
                best.answer = self._coerce_mcq_answer(best.answer)
            return best
        result = self._run_once(query)
        if self.verify and self._model_calls < self.max_model_calls:
            verified = self._verify_answer(query, result.answer)
            result.answer = verified.answer
            result.checks = verified.checks
            result.confidence = verified.confidence
        if self._mcq_letters:
            result.answer = self._coerce_mcq_answer(result.answer)
        return result

    def _run_once(self, query: str, nonce: str | None = None) -> AgentResult:
        logger.info("New agent run (nonce=%s).", nonce or "none")
        logger.info("User query: %s", redact(query))
        self.tools_used = []
        self.tools_created = []
        self.memory = MemoryStore(
            max_tool_output_chars=self.memory.max_tool_output_chars,
            keep_raw_tool_output=self.memory.keep_raw_tool_output,
            summary_lines=self.memory.summary_lines,
        )
        self._messages = []
        system_prompt = (
            "You are an agent that can use tools. "
            "Use tools when necessary. "
            "Respond with JSON only, using either "
            '{"type":"tool","name":"<tool>","arguments":{...}} '
            'or {"type":"final","answer":"...","confidence":0.0,"checks":[...]} '
            "Do not reveal chain-of-thought.\n"
            "Examples:\n"
            'Tool: {"type":"tool","name":"calculator","arguments":{"expression":"2+2"}}\n'
            'Final: {"type":"final","answer":"4","confidence":0.9,"checks":["basic arithmetic"]}'
        )
        if self.strict_json_mode:
            system_prompt += (
                " Output must be exactly one JSON object and nothing else."
            )
        self._append_message({"role": "system", "content": system_prompt})
        if nonce:
            self._append_message({"role": "system", "content": f"Nonce: {nonce}"})
        self._append_message({"role": "user", "content": query})

        if self.mode == "deep":
            self._internal_plan(query)

        if self.trace:
            self.trace.record_messages(self._messages)

        tools_enabled, tools_reason = should_enable_tools(query)
        if self._mcq_letters:
            tools_enabled = False
            tools_reason = "mcq"
        if not tools_enabled and not self._mcq_letters:
            for tool in self.registry.list():
                if re.search(rf"\b{re.escape(tool.name)}\b", query, re.IGNORECASE):
                    tools_enabled = True
                    tools_reason = "explicit_tool_name"
                    break
        if not tools_enabled:
            logger.info("Controller disabled tools (%s).", tools_reason)
            self._append_message(
                {
                    "role": "system",
                    "content": (
                        "Controller decision: tools disabled. Answer directly without "
                        "calling tools."
                    ),
                }
            )
        elif tools_reason == "dynamic_info":
            logger.info("Controller enabled tools (dynamic_info).")
        elif tools_reason == "deterministic_compute":
            logger.info("Controller enabled tools (deterministic_compute).")

        task_type = determine_task_type(query, tools_reason)
        logger.info("Task type determined: %s", task_type.name)

        special_case = (
            self.verify
            or self.self_consistency > 1
            or self.code_check
            or self._has_strict_format_contract(query)
        )
        if (
            not tools_enabled
            and tools_reason == "short_closed_book"
            and not special_case
        ):
            fast_path = self._run_fast_path_final(query)
            if fast_path is not None:
                return fast_path

        remaining_tool_calls = max(0, self.max_tool_calls - self._tool_calls)
        remaining_model_calls = max(0, self.max_model_calls - self._model_calls)
        logger.info(
            "Limits: model_calls=%s, tool_calls=%s",
            remaining_model_calls,
            remaining_tool_calls,
        )
        autopilot = self._maybe_autopilot_tool(
            query,
            tools_enabled=tools_enabled,
            tools_reason=tools_reason,
            remaining_tool_calls=remaining_tool_calls,
        )
        if autopilot is not None:
            return autopilot

        problem_state = self._build_problem_state(query)
        logger.info("Problem state initialized: %s", redact(self._state_to_json(problem_state)))

        return self._run_reflective_loop(
            query,
            problem_state,
            task_type=task_type,
            tools_enabled=tools_enabled,
            tools_reason=tools_reason,
            remaining_model_calls=remaining_model_calls,
            remaining_tool_calls=remaining_tool_calls,
        )

    def _run_fast_path_final(self, query: str) -> AgentResult | None:
        if self._model_calls >= self.max_model_calls:
            return AgentResult(
                answer="Reached model call limit",
                tools_used=self.tools_used,
                tools_created=self.tools_created,
                trace_path=self._finalize_trace(),
            )
        response = self.model.chat(self._messages, tools=None)
        self._model_calls += 1
        if self.trace:
            tool_payload = (
                response.tool_call.model_dump()
                if response.tool_call is not None
                else None
            )
            self.trace.record_model_response(response.final_text, tool_payload)
        protocol = None
        if response.final_text:
            protocol = self._parse_model_protocol(response.final_text)
        if response.tool_call is None and response.final_text:
            tool_call = self._parse_tool_call_from_text(response.final_text)
            if tool_call is not None:
                response = response.model_copy(update={"tool_call": tool_call})
        if response.tool_call is not None or isinstance(protocol, ProtocolToolCall):
            self._pending_model_response = response
            return None
        if isinstance(protocol, ProtocolFinal):
            answer = format_final(protocol.answer, protocol.checks)
        else:
            answer = response.final_text or ""
        answer = answer.strip()
        if not answer:
            answer = "No response from model."
        return AgentResult(
            answer=answer,
            tools_used=self.tools_used,
            tools_created=self.tools_created,
            trace_path=self._finalize_trace(),
        )

    def _build_problem_state(self, query: str) -> ProblemState:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are analyzing a problem. Identify constraints, known facts, "
                    "unknowns, and a high-level plan. Do not solve yet. "
                    "Return JSON with keys: constraints, known_facts, unknowns, plan."
                ),
            },
            {"role": "user", "content": query},
        ]
        if self._model_calls >= self.max_model_calls:
            return ProblemState(objective=query)
        response = self.model.chat(messages, tools=None)
        self._model_calls += 1
        protocol = None
        if response.final_text:
            protocol = self._parse_model_protocol(response.final_text)
        if response.tool_call is not None:
            self._pending_model_response = response
            return ProblemState(objective=query)
        payload = self._parse_json_payload(response.final_text or "")
        if not isinstance(payload, dict):
            if isinstance(protocol, ProtocolToolCall):
                if not self.policy.red_flag_strict_json or response.final_text.lstrip().startswith("{"):
                    self._pending_model_response = response
            elif isinstance(protocol, ProtocolFinal):
                self._pending_model_response = response
            return ProblemState(objective=query)
        expected_keys = {"constraints", "known_facts", "unknowns", "plan"}
        if not expected_keys.intersection(payload.keys()):
            payload_type = payload.get("type") if isinstance(payload, dict) else None
            if isinstance(protocol, ProtocolToolCall) or payload_type == "tool":
                if not self.policy.red_flag_strict_json or response.final_text.lstrip().startswith("{"):
                    self._pending_model_response = response
            elif isinstance(protocol, ProtocolFinal) or payload_type == "final":
                self._pending_model_response = response
            return ProblemState(objective=query)
        assumed_facts = [
            Fact(content=str(item), source=EpistemicSource.ASSUMED, provenance="initial_analysis")
            for item in payload.get("known_facts") or []
        ]
        return ProblemState(
            objective=query,
            constraints=[str(item) for item in payload.get("constraints") or []],
            known_facts=assumed_facts,
            unknowns=[str(item) for item in payload.get("unknowns") or []],
            plan=[str(item) for item in payload.get("plan") or []],
        )

    def _run_reflective_loop(
        self,
        query: str,
        problem_state: ProblemState,
        *,
        task_type: TaskType,
        tools_enabled: bool,
        tools_reason: str,
        remaining_model_calls: int,
        remaining_tool_calls: int,
    ) -> AgentResult:
        candidates: list[tuple[str, ProblemState]] = []
        code_check_enabled = self.code_check and is_code_task(query)
        used_router = False
        short_task = tools_reason == "short_closed_book"
        ambiguous_tool_vote = (
            tools_enabled
            and self.policy.tool_vote_enabled
            and is_route_ambiguous(tool_candidates(query))
        )
        if tools_enabled:
            suggestion = suggest_tool(query)
            if suggestion and not used_router:
                logger.info(
                    "Router suggested tool: %s (%s)",
                    suggestion.tool_name,
                    suggestion.reason,
                )
                self._append_message(
                    {
                        "role": "system",
                        "content": f"Router suggestion: {suggestion.tool_name} ({suggestion.reason}).",
                    }
                )
                used_router = True
        for step_index in range(self.max_steps):
            if remaining_model_calls <= 0:
                return AgentResult(
                    answer="Reached model call limit",
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    trace_path=self._finalize_trace(),
                )
            step_desc = problem_state.plan[0] if problem_state.plan else "Advance the objective."
            outcome = self._execute_plan_step(
                problem_state,
                step_desc,
                tools_enabled=tools_enabled,
                remaining_tool_calls=remaining_tool_calls,
                remaining_model_calls=remaining_model_calls,
                ambiguous_tool_vote=ambiguous_tool_vote,
            )
            remaining_tool_calls = max(0, self.max_tool_calls - self._tool_calls)
            remaining_model_calls = max(0, self.max_model_calls - self._model_calls)
            if outcome.fact is not None:
                problem_state.known_facts.append(outcome.fact)
            if outcome.candidate_solution:
                problem_state.candidate_solution = outcome.candidate_solution
                candidates.append((outcome.candidate_solution, problem_state))
            if outcome.failure_reason:
                problem_state.failure_reason = outcome.failure_reason

            reflection = self._reflect_on_step(problem_state, outcome.step_summary)
            if reflection is not None:
                advanced = reflection.get("advanced", True)
                reason = reflection.get("reason")
                revised_plan = reflection.get("plan")
                logger.info(
                    "Reflection outcome: advanced=%s reason=%s",
                    advanced,
                    redact(str(reason) if reason is not None else ""),
                )
                if reflection.get("known_facts"):
                    inferred = self._parse_inferred_facts(reflection.get("known_facts"))
                    problem_state.known_facts.extend(inferred)
                if reflection.get("unknowns") is not None:
                    problem_state.unknowns = [
                        str(item) for item in reflection.get("unknowns") or []
                    ]
                if not advanced:
                    problem_state.failure_reason = reason or problem_state.failure_reason
                    if isinstance(revised_plan, list) and revised_plan:
                        problem_state.plan = [str(item) for item in revised_plan]
                        logger.info("Plan revised: %s", redact(json.dumps(problem_state.plan)))
                    logger.info("Reflection indicated lack of progress: %s", redact(reason or ""))
            if problem_state.candidate_solution and not problem_state.unknowns:
                epistemic_ok = self._ensure_epistemic_validity(
                    problem_state,
                    task_type,
                    remaining_model_calls=remaining_model_calls,
                    remaining_tool_calls=remaining_tool_calls,
                )
                if epistemic_ok:
                    break
                if remaining_model_calls <= 0 and remaining_tool_calls <= 0:
                    return AgentResult(
                        answer="Cannot justify answer with acceptable evidence.",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                problem_state.unknowns = ["Need additional evidence for epistemic justification."]
                problem_state.candidate_solution = None
            if (
                step_index == 0
                and not problem_state.candidate_solution
                and problem_state.unknowns
                and remaining_model_calls >= 4
            ):
                alternative_paths = self._generate_alternative_paths(problem_state, count=2)
                for idx, alt_state in enumerate(alternative_paths):
                    logger.info("Exploring alternative path %s: %s", idx + 1, redact(json.dumps(alt_state.plan)))
                    alt_outcome = self._execute_alternative_step(query, alt_state)
                    if alt_outcome.fact is not None:
                        alt_state.known_facts.append(alt_outcome.fact)
                    if alt_outcome.candidate_solution:
                        alt_state.candidate_solution = alt_outcome.candidate_solution
                        candidates.append((alt_outcome.candidate_solution, alt_state))

        if not candidates and problem_state.candidate_solution:
            candidates.append((problem_state.candidate_solution, problem_state))

        if not candidates:
            return AgentResult(
                answer="Unable to derive a solution with available information.",
                tools_used=self.tools_used,
                tools_created=self.tools_created,
                trace_path=self._finalize_trace(),
            )

        verified = self._verify_candidates(problem_state, candidates)
        if verified is None:
            best_solution, best_state = candidates[0]
        else:
            best_solution, best_state = verified

        epistemic_ok = self._ensure_epistemic_validity(
            best_state,
            task_type,
            remaining_model_calls=remaining_model_calls,
            remaining_tool_calls=remaining_tool_calls,
        )
        if not epistemic_ok:
            return AgentResult(
                answer="Cannot justify answer with acceptable evidence.",
                tools_used=self.tools_used,
                tools_created=self.tools_created,
                trace_path=self._finalize_trace(),
            )

        if remaining_model_calls >= 2:
            confidence, confidence_reason = self._estimate_confidence(best_state, best_solution)
        else:
            confidence = 0.7
            confidence_reason = "Confidence estimate skipped due to limited model calls."
        best_state.confidence = confidence

        final_answer = best_solution
        if code_check_enabled:
            final_answer = self._code_check_loop(final_answer)
        if self._mcq_letters:
            final_answer = self._coerce_mcq_answer(final_answer)
        if confidence < 0.6:
            if short_task:
                checks = []
                if confidence_reason:
                    checks.append(f"Low confidence: {confidence_reason}")
                else:
                    checks.append("Low confidence")
                return AgentResult(
                    answer=final_answer,
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    checks=checks,
                    confidence=confidence,
                    trace_path=self._finalize_trace(),
                )
            refusal = "I am not confident enough to provide a definitive answer."
            if confidence_reason:
                refusal = f"{refusal} {confidence_reason}"
            return AgentResult(
                answer=refusal,
                tools_used=self.tools_used,
                tools_created=self.tools_created,
                confidence=confidence,
                trace_path=self._finalize_trace(),
            )
        return AgentResult(
            answer=final_answer,
            tools_used=self.tools_used,
            tools_created=self.tools_created,
            confidence=confidence,
            trace_path=self._finalize_trace(),
        )

    def _execute_plan_step(
        self,
        problem_state: ProblemState,
        step_desc: str,
        *,
        tools_enabled: bool,
        remaining_tool_calls: int,
        remaining_model_calls: int,
        ambiguous_tool_vote: bool,
    ) -> StepOutcome:
        self._append_message(
            {
                "role": "system",
                "content": (
                    "Execute the next plan step. Use tools if necessary. "
                    f"Plan step: {step_desc}"
                ),
            }
        )
        tools = self.registry.openai_schemas() if tools_enabled else None
        if self._model_calls >= self.max_model_calls:
            return StepOutcome(step_summary="Model call limit reached.", failure_reason="limit")
        if self._pending_model_response is not None:
            response = self._pending_model_response
            self._pending_model_response = None
        else:
            response = self.model.chat(self._messages, tools=tools)
            self._model_calls += 1
        if self.trace:
            tool_payload = (
                response.tool_call.model_dump()
                if response.tool_call is not None
                else None
            )
            self.trace.record_model_response(response.final_text, tool_payload)
        protocol = None
        if response.final_text:
            protocol = self._parse_model_protocol(response.final_text)
        if response.tool_call is None and response.final_text:
            tool_call = self._parse_tool_call_from_text(response.final_text)
            if tool_call is not None:
                response = response.model_copy(update={"tool_call": tool_call})
        if tools_enabled and ambiguous_tool_vote:
            elected, remaining_model_calls = self._maybe_vote_tool_call(
                tools,
                response,
                remaining_model_calls,
                ambiguous=True,
            )
            if elected is not None:
                response = response.model_copy(update={"tool_call": elected})
        if response.tool_call is None:
            if isinstance(protocol, ProtocolFinal):
                answer = format_final(protocol.answer, protocol.checks)
            else:
                answer = response.final_text or ""
            candidate = answer.strip()
            if not candidate:
                return StepOutcome(
                    step_summary="Model returned no answer for the plan step.",
                    failure_reason="no_answer",
                )
            return StepOutcome(
                step_summary="Derived candidate solution from reasoning.",
                candidate_solution=candidate,
            )
        if not tools_enabled:
            return StepOutcome(
                step_summary="Tool call requested while tools are disabled.",
                failure_reason="tools_disabled",
            )
        if remaining_tool_calls <= 0:
            return StepOutcome(
                step_summary="Tool call limit reached.",
                failure_reason="tool_limit",
            )
        tool_name = response.tool_call.name
        tool = self.registry.get(tool_name)
        if tool is None or not self.policy.is_tool_allowed(tool_name):
            return StepOutcome(
                step_summary=f"Requested unknown or disallowed tool: {tool_name}",
                failure_reason="tool_unavailable",
            )
        try:
            validated_args = tool.input_schema.model_validate(
                response.tool_call.arguments
            ).model_dump()
        except ValidationError as exc:
            elected, remaining_model_calls = self._maybe_vote_tool_call(
                tools,
                response,
                remaining_model_calls,
                schema_error=True,
            )
            if elected is not None:
                tool = self.registry.get(elected.name)
                if tool is None or not self.policy.is_tool_allowed(elected.name):
                    return StepOutcome(
                        step_summary=(
                            "Tool vote elected an unknown or disallowed tool "
                            f"{elected.name}."
                        ),
                        failure_reason="tool_unavailable",
                    )
                output, ok, _schema_error = self._execute_tool(tool, elected.arguments)
                self._handle_tool_result(
                    tool.name,
                    output,
                    response.tool_call.id if response.tool_call else None,
                    elected.arguments,
                )
                self._tool_calls += 1
                summary = f"Tool {tool.name} returned: {self.memory.entries[-1].summary}"
                fact = Fact(
                    content=self.memory.entries[-1].summary,
                    source=self._tool_epistemic_source(tool.name),
                    provenance=tool.name,
                )
                if ok is False:
                    return StepOutcome(
                        step_summary=summary,
                        failure_reason="tool_failed",
                        fact=fact,
                    )
                return StepOutcome(
                    step_summary=f"Recovered tool call via voting: {summary}",
                    fact=fact,
                )
            output = self._tool_error_payload(tool, exc, schema_error=True)
            self._handle_tool_result(
                tool_name,
                output,
                response.tool_call.id,
                response.tool_call.arguments,
            )
            return StepOutcome(
                step_summary=f"Tool schema error for {tool_name}.",
                failure_reason="tool_schema_error",
            )
        logger.info(
            "Tool call requested: %s args=%s",
            tool_name,
            redact(json.dumps(validated_args, ensure_ascii=False)),
        )
        output, ok, _schema_error = self._execute_tool(tool, validated_args)
        self._handle_tool_result(tool_name, output, response.tool_call.id, validated_args)
        self._tool_calls += 1
        summary = f"Tool {tool_name} returned: {self.memory.entries[-1].summary}"
        fact = Fact(
            content=self.memory.entries[-1].summary,
            source=self._tool_epistemic_source(tool_name),
            provenance=tool_name,
        )
        if ok is False:
            return StepOutcome(step_summary=summary, failure_reason="tool_failed", fact=fact)
        return StepOutcome(step_summary=summary, fact=fact)

    def _reflect_on_step(
        self,
        problem_state: ProblemState,
        step_summary: str,
    ) -> dict[str, Any] | None:
        if self._model_calls >= self.max_model_calls:
            return None
        messages = [
            {
                "role": "system",
                "content": (
                    "Given the current problem state, did the last step reduce "
                    "uncertainty or advance the solution? If not, explain why and "
                    "revise the plan. Return JSON with keys: advanced, reason, plan, "
                    "known_facts, unknowns. When adding known_facts, return objects "
                    "with 'content' and optional 'provenance' that reference earlier facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem state: {self._state_to_json(problem_state)}\n"
                    f"Last step: {step_summary}"
                ),
            },
        ]
        response = self.model.chat(messages, tools=None)
        self._model_calls += 1
        payload = self._parse_json_payload(response.final_text or "")
        if not isinstance(payload, dict):
            if response.final_text:
                protocol = self._parse_model_protocol(response.final_text)
                if isinstance(protocol, (ProtocolFinal, ProtocolToolCall)):
                    self._pending_model_response = response
            return None
        expected_keys = {"advanced", "reason", "plan", "known_facts", "unknowns"}
        if not expected_keys.intersection(payload.keys()):
            if response.final_text:
                protocol = self._parse_model_protocol(response.final_text)
                if isinstance(protocol, (ProtocolFinal, ProtocolToolCall)):
                    self._pending_model_response = response
            return None
        return payload

    def _generate_alternative_paths(
        self, problem_state: ProblemState, *, count: int
    ) -> list[ProblemState]:
        if self._model_calls >= self.max_model_calls:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate alternative solution paths with different approaches. "
                    "Return JSON with a 'paths' list, each containing a 'plan' list."
                ),
            },
            {
                "role": "user",
                "content": self._state_to_json(problem_state),
            },
        ]
        response = self.model.chat(messages, tools=None)
        self._model_calls += 1
        payload = self._parse_json_payload(response.final_text or "")
        paths = []
        for item in (payload or {}).get("paths", [])[:count]:
            plan = [str(step) for step in item.get("plan") or []]
            paths.append(replace(problem_state, plan=plan, candidate_solution=None))
        return paths

    def _execute_alternative_step(self, query: str, problem_state: ProblemState) -> StepOutcome:
        if self._model_calls >= self.max_model_calls:
            return StepOutcome(step_summary="Model call limit reached.", failure_reason="limit")
        step_desc = problem_state.plan[0] if problem_state.plan else "Advance the objective."
        messages = [
            {
                "role": "system",
                "content": (
                    "Execute one plan step using a distinct reasoning approach. "
                    "Return either a final answer or a concise summary."
                ),
            },
            {"role": "user", "content": f"Objective: {query}\nPlan step: {step_desc}"},
        ]
        response = self.model.chat(messages, tools=None)
        self._model_calls += 1
        text = response.final_text or ""
        candidate = text.strip()
        if not candidate:
            return StepOutcome(
                step_summary="Alternative path produced no response.",
                failure_reason="no_response",
            )
        return StepOutcome(
            step_summary="Alternative path produced a candidate solution.",
            candidate_solution=candidate,
        )

    def _verify_candidates(
        self,
        problem_state: ProblemState,
        candidates: list[tuple[str, ProblemState]],
    ) -> tuple[str, ProblemState] | None:
        if self._model_calls >= self.max_model_calls:
            return None
        payload = {
            "objective": problem_state.objective,
            "constraints": problem_state.constraints,
            "known_facts": [self._fact_to_payload(fact) for fact in problem_state.known_facts],
            "candidates": [
                {"index": idx, "solution": solution}
                for idx, (solution, _state) in enumerate(candidates)
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Verify each candidate solution against the original objective and "
                    "constraints. Identify inconsistencies, missing assumptions, or "
                    "contradictions. Return JSON with a 'results' list containing "
                    "index, valid, issues, unknowns."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = self.model.chat(messages, tools=None)
        self._model_calls += 1
        verification = self._parse_json_payload(response.final_text or "")
        results = (verification or {}).get("results") or []
        best_idx = None
        best_score = None
        for result in results:
            if not isinstance(result, dict):
                continue
            idx = result.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
                continue
            valid = result.get("valid", True)
            if not valid:
                continue
            unknowns = result.get("unknowns") or []
            score = len(unknowns)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            return None
        return candidates[best_idx]

    def _estimate_confidence(
        self, problem_state: ProblemState, solution: str
    ) -> tuple[float, str | None]:
        if self._model_calls >= self.max_model_calls:
            return 0.7, "Confidence could not be estimated."
        messages = [
            {
                "role": "system",
                "content": (
                    "Estimate confidence (0–1) in the selected solution. "
                    "If confidence < 0.6, explain why. Return JSON with keys: "
                    "confidence, reason."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Objective: {problem_state.objective}\n"
                    f"Constraints: {problem_state.constraints}\n"
                    f"Known facts: {[self._fact_to_payload(fact) for fact in problem_state.known_facts]}\n"
                    f"Solution: {solution}"
                ),
            },
        ]
        response = self.model.chat(messages, tools=None)
        self._model_calls += 1
        payload = self._parse_json_payload(response.final_text or "")
        if not isinstance(payload, dict):
            return 0.7, "Confidence could not be estimated."
        confidence = payload.get("confidence", 0.0)
        reason = payload.get("reason")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.7
        return confidence_value, str(reason) if reason else None

    def _parse_json_payload(self, content: str) -> dict[str, Any] | None:
        if not content:
            return None
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            try:
                payload = repair_json(content)
            except JsonRepairError:
                return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _tool_epistemic_source(self, tool_name: str) -> EpistemicSource:
        observed_tools = {"http_fetch", "filesystem", "web_search"}
        derived_tools = {
            "calculator",
            "python_sandbox",
            "code_run_multi",
            "unit_convert",
            "json_repair",
            "regex_extract",
        }
        if tool_name in observed_tools or re.search(r"(fetch|search|scrape)", tool_name):
            return EpistemicSource.OBSERVED
        if tool_name in derived_tools:
            return EpistemicSource.DERIVED
        return EpistemicSource.DERIVED

    def _collect_answer_facts(self, problem_state: ProblemState, solution: str) -> list[Fact]:
        if self._model_calls >= self.max_model_calls:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "List the minimal facts required to justify the answer. "
                    "For each fact, specify whether it is assumed, observed, "
                    "derived, inferred, or validated. Return JSON with key "
                    "'facts' as a list of objects with content, source, provenance."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Known facts: {[self._fact_to_payload(fact) for fact in problem_state.known_facts]}\n"
                    f"Answer: {solution}"
                ),
            },
        ]
        response = self.model.chat(messages, tools=None)
        self._model_calls += 1
        payload = self._parse_json_payload(response.final_text or "")
        items = (payload or {}).get("facts") or []
        facts: list[Fact] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            source = item.get("source")
            provenance = item.get("provenance")
            if not content or not source:
                continue
            source_enum = self._normalize_source(str(source))
            if source_enum is None:
                continue
            facts.append(
                Fact(
                    content=str(content),
                    source=source_enum,
                    provenance=str(provenance) if provenance else None,
                )
            )
        return facts

    def _fallback_answer_facts(self, solution: str, known_facts: list[Fact]) -> list[Fact]:
        if known_facts:
            return list(known_facts)
        if solution.strip():
            return [
                Fact(
                    content=solution.strip(),
                    source=EpistemicSource.INFERRED,
                    provenance="fallback_from_answer",
                )
            ]
        return []

    def _normalize_source(self, source: str) -> EpistemicSource | None:
        normalized = source.strip().upper()
        for candidate in EpistemicSource:
            if candidate.name == normalized:
                return candidate
        return None

    def _validate_extraction_facts(
        self, answer_facts: list[Fact], known_facts: list[Fact]
    ) -> list[Fact]:
        validated: list[Fact] = []
        known_contents = [fact.content for fact in known_facts]
        for fact in answer_facts:
            if any(fact.content in known for known in known_contents):
                validated.append(
                    Fact(
                        content=fact.content,
                        source=EpistemicSource.VALIDATED,
                        provenance="validated_by_substring",
                    )
                )
        return validated

    def _ensure_epistemic_validity(
        self,
        problem_state: ProblemState,
        task_type: TaskType,
        *,
        remaining_model_calls: int,
        remaining_tool_calls: int,
    ) -> bool:
        solution = problem_state.candidate_solution or ""
        if remaining_model_calls > 2:
            answer_facts = self._collect_answer_facts(problem_state, solution)
        else:
            answer_facts = []
        if not answer_facts:
            answer_facts = self._fallback_answer_facts(solution, problem_state.known_facts)
        if task_type == TaskType.EXTRACTION:
            answer_facts = self._validate_extraction_facts(answer_facts, problem_state.known_facts)
        problem_state.answer_facts = answer_facts
        allowed_sources = EPISTEMIC_REQUIREMENTS.get(task_type, set())
        if not answer_facts:
            self._log_epistemic_summary(problem_state, task_type, passed=False)
            return False
        for fact in answer_facts:
            if fact.source not in allowed_sources:
                self._log_epistemic_summary(problem_state, task_type, passed=False)
                return False
        if task_type == TaskType.DETERMINISTIC_COMPUTE:
            if not any(fact.source == EpistemicSource.DERIVED for fact in answer_facts):
                self._log_epistemic_summary(problem_state, task_type, passed=False)
                return False
        if task_type == TaskType.EXTRACTION:
            if not answer_facts or any(
                fact.source != EpistemicSource.VALIDATED for fact in answer_facts
            ):
                self._log_epistemic_summary(problem_state, task_type, passed=False)
                return False
        self._log_epistemic_summary(problem_state, task_type, passed=True)
        return True

    def _log_epistemic_summary(
        self, problem_state: ProblemState, task_type: TaskType, *, passed: bool
    ) -> None:
        known_payload = [self._fact_to_payload(fact) for fact in problem_state.known_facts]
        answer_payload = [self._fact_to_payload(fact) for fact in problem_state.answer_facts]
        logger.info("Epistemic summary: task_type=%s", task_type.name)
        logger.info("Epistemic known facts: %s", redact(json.dumps(known_payload, ensure_ascii=False)))
        logger.info(
            "Epistemic answer facts: %s",
            redact(json.dumps(answer_payload, ensure_ascii=False)),
        )
        logger.info("Epistemic justification: %s", "PASS" if passed else "FAIL")

    def _state_to_json(self, problem_state: ProblemState) -> str:
        payload = {
            "objective": problem_state.objective,
            "constraints": problem_state.constraints,
            "known_facts": [self._fact_to_payload(fact) for fact in problem_state.known_facts],
            "unknowns": problem_state.unknowns,
            "plan": problem_state.plan,
            "failure_reason": problem_state.failure_reason,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _fact_to_payload(self, fact: Fact) -> dict[str, Any]:
        return {
            "content": fact.content,
            "source": fact.source.name,
            "provenance": fact.provenance,
        }

    def _parse_inferred_facts(self, items: Any) -> list[Fact]:
        facts: list[Fact] = []
        if not isinstance(items, list):
            return facts
        for item in items:
            if isinstance(item, dict):
                content = item.get("content")
                if content is None:
                    continue
                provenance = item.get("provenance")
                facts.append(
                    Fact(
                        content=str(content),
                        source=EpistemicSource.INFERRED,
                        provenance=str(provenance) if provenance else "reflection",
                    )
                )
            else:
                facts.append(
                    Fact(
                        content=str(item),
                        source=EpistemicSource.INFERRED,
                        provenance="reflection",
                    )
                )
        return facts

    def _handle_tool_result(
        self,
        tool_name: str,
        output: Any,
        call_id: str | None = None,
        arguments: dict[str, Any] | None = None,
        message_role: str = "tool",
        explanation: str | None = None,
    ) -> None:
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        if tool_name == "tool_maker":
            tool_created = output.get("tool") if isinstance(output, dict) else None
            if tool_created:
                self.tools_created.append(tool_created)
        entry = self.memory.add_tool_output(tool_name, output)
        payload = {
            "handle": entry.handle,
            "summary": entry.summary,
        }
        content = json.dumps(payload, ensure_ascii=False, default=str)
        if message_role == "tool":
            tool_message = {
                "role": "tool",
                "tool_call_id": call_id or tool_name,
                "content": content,
            }
        else:
            note = explanation or f"Auto-executed tool {tool_name}."
            tool_message = {
                "role": message_role,
                "content": f"{note}\n{entry.summary}",
            }
        self._append_message(tool_message)
        logger.info(
            "Tool result stored: %s summary=%s",
            tool_name,
            redact(entry.summary),
        )
        if self.trace:
            if arguments is not None:
                self.trace.record_tool_call(tool_name, arguments)
            self.trace.record_tool_result(tool_name, entry.handle, entry.summary)

    def _prepare_tool_output(self, output: Any) -> Any:
        if output is None:
            return None
        budget = self.memory.max_tool_output_chars
        try:
            serialized = (
                output
                if isinstance(output, str)
                else json.dumps(output, ensure_ascii=False, default=str)
            )
        except TypeError:
            serialized = str(output)
        total_chars = len(serialized)
        if total_chars <= budget:
            return output
        head_len = budget // 2
        tail_len = budget - head_len
        return {
            "truncated": True,
            "total_chars": total_chars,
            "head": serialized[:head_len],
            "tail": serialized[-tail_len:] if tail_len > 0 else "",
        }

    def _direct_tool_args(self, tool_name: str, query: str) -> dict[str, Any] | None:
        if tool_name == "calculator":
            matches = re.findall(r"[0-9\.\s\+\-\*\/\(\)]+", query)
            if matches:
                expression = max(matches, key=len).strip()
                if expression:
                    return {"expression": expression}
            return {"expression": query.strip()}
        if tool_name == "json_repair":
            return {"text": query}
        if tool_name == "http_fetch":
            match = re.search(r"https?://\S+", query)
            if match:
                return {"url": match.group(0)}
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

    def _maybe_autopilot_tool(
        self,
        query: str,
        *,
        tools_enabled: bool,
        tools_reason: str,
        remaining_tool_calls: int,
    ) -> AgentResult | None:
        if not tools_enabled or remaining_tool_calls <= 0:
            return None
        deterministic_tools = {"calculator", "unit_convert", "json_repair"}
        tool_name: str | None = None
        if tools_reason in {"math", "unit", "json"}:
            tool_name = {
                "math": "calculator",
                "unit": "unit_convert",
                "json": "json_repair",
            }.get(tools_reason)
        if tool_name is None:
            suggestion = suggest_tool(query)
            if (
                suggestion
                and suggestion.confidence >= 0.75
                and suggestion.tool_name in deterministic_tools
            ):
                tool_name = suggestion.tool_name
        if tool_name is None or tool_name not in deterministic_tools:
            return None
        tool = self.registry.get(tool_name)
        if tool is None or not self.policy.is_tool_allowed(tool_name):
            return None
        args = self._direct_tool_args(tool_name, query)
        if args is None:
            return None
        try:
            validated_args = tool.input_schema.model_validate(args).model_dump()
        except ValidationError:
            return None
        output, ok, _schema_error = self._execute_tool(tool, validated_args)
        if ok is False:
            return None
        if tool_name == "json_repair" and isinstance(output, dict) and output.get("error"):
            return None
        self._handle_tool_result(
            tool_name,
            output,
            arguments=validated_args,
            message_role="system",
            explanation="Auto-executed deterministic tool.",
        )
        self._tool_calls += 1
        if tool_name == "calculator":
            answer = str((output or {}).get("value", "")).strip()
        elif tool_name == "unit_convert":
            value = (output or {}).get("value")
            unit = (output or {}).get("unit")
            answer = str(value)
            if unit:
                answer = f"{answer} {unit}"
        else:
            value = (output or {}).get("value")
            if isinstance(value, (dict, list)):
                answer = json.dumps(value, ensure_ascii=False)
            else:
                answer = str(value)
        answer = answer.strip() or "No result from tool."
        return AgentResult(
            answer=answer,
            tools_used=self.tools_used,
            tools_created=self.tools_created,
            trace_path=self._finalize_trace(),
        )

    def _has_strict_format_contract(self, query: str) -> bool:
        patterns = [r"\breturn\b", r"format:", r"answer only", r"answer exactly"]
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns)

    def _format_contract_violation(self, query: str, answer: str) -> bool:
        lower = query.lower()
        expects_csv = "csv" in lower or "comma-separated" in lower or "comma separated" in lower
        expects_table = "table" in lower or "tabular" in lower
        expects_pipe = "pipe-separated" in lower or "pipe separated" in lower or "|" in query
        has_sentence = self._looks_like_sentence(answer)
        if (expects_csv or expects_table) and has_sentence:
            return True
        if expects_csv and "," not in answer:
            return True
        if expects_pipe and "|" not in answer:
            return True
        if expects_table and ("\n" not in answer and "|" not in answer and "\t" not in answer):
            return True
        return False

    def _looks_like_sentence(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped.split()) < 4:
            return False
        return bool(re.search(r"[.!?](\s|$)", stripped))

    def _is_logic_yes_no_task(self, query: str) -> bool:
        return bool(
            re.search(r"can we conclude", query, re.IGNORECASE)
            and re.search(r"answer yes or no only", query, re.IGNORECASE)
        )

    def _pick_best(self, candidates: list[AgentResult]) -> AgentResult:
        def score(candidate: AgentResult) -> tuple[float, int, int]:
            confidence = candidate.confidence or 0.0
            check_len = len(candidate.checks)
            answer_len = len(candidate.answer)
            if confidence <= 0:
                return (confidence, -answer_len, -check_len)
            return (confidence, check_len, -answer_len)

        return max(candidates, key=score)

    def _pick_majority_mcq(self, candidates: list[AgentResult]) -> AgentResult:
        letter_votes: dict[str, int] = {}
        letter_candidates: dict[str, list[AgentResult]] = {}
        for candidate in candidates:
            letter = self._extract_mcq_letter(candidate.answer)
            if not letter:
                continue
            letter_votes[letter] = letter_votes.get(letter, 0) + 1
            letter_candidates.setdefault(letter, []).append(candidate)
        if not letter_votes:
            return self._pick_best(candidates)
        top_count = max(letter_votes.values())
        top_letters = [letter for letter, count in letter_votes.items() if count == top_count]
        if len(top_letters) == 1:
            letter = top_letters[0]
            best = self._pick_best(letter_candidates[letter])
            best.answer = self._coerce_mcq_answer(letter)
            return best
        tied_candidates = [cand for letter in top_letters for cand in letter_candidates[letter]]
        best = self._pick_best(tied_candidates)
        best.answer = self._coerce_mcq_answer(best.answer)
        return best

    def _verify_answer(self, query: str, answer: str) -> AgentResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "Verify the answer. Respond with JSON only using "
                    '{"type":"final","answer":"...","confidence":0.0,"checks":[...]} '
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
                return AgentResult(
                    answer=format_final(protocol.answer, protocol.checks),
                    checks=protocol.checks,
                    confidence=protocol.confidence,
                )
        return AgentResult(answer=answer)

    def _finalize_trace(self) -> str | None:
        if not self.trace:
            return None
        stats = {
            "model_calls": self._model_calls,
            "tool_calls": self._tool_calls,
        }
        return self.trace.finalize(stats)

    def _execute_tool(
        self, tool: Tool, arguments: dict[str, Any]
    ) -> tuple[Any, bool, bool]:
        try:
            result = tool.run(arguments)
            output = result.output
            ok = not (isinstance(output, dict) and output.get("ok") is False)
            return output, ok, False
        except ValidationError as exc:
            return self._tool_error_payload(tool, exc, schema_error=True), False, True
        except (ValueError, TypeError, KeyError) as exc:
            return self._tool_error_payload(tool, exc), False, False
        except Exception as exc:  # noqa: BLE001
            return self._tool_error_payload(tool, exc), False, False

    def _tool_error_payload(
        self, tool: Tool, exc: Exception, schema_error: bool = False
    ) -> dict[str, Any]:
        schema = None
        example = None
        if getattr(tool, "input_schema", None) is not None:
            schema = tool.input_schema.model_json_schema()
            if schema_error:
                example = self._schema_example(schema, tool.name)
        return {
            "ok": False,
            "tool": tool.name,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "expected_input_schema": schema,
            "example": example,
            "hint": "Fix arguments and retry",
        }

    def _schema_example(self, schema: dict[str, Any], tool_name: str | None = None) -> dict[str, Any]:
        if tool_name == "python_sandbox":
            return {"code": "print(1)", "timeout_seconds": 2}
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        example: dict[str, Any] = {}
        for name in required:
            example[name] = self._example_value(properties.get(name, {}))
        return example

    def _example_value(self, schema: dict[str, Any]) -> Any:
        if "default" in schema:
            return schema["default"]
        examples = schema.get("examples")
        if examples:
            return examples[0]
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            schema_type = schema_type[0]
        if schema_type == "string":
            return "value"
        if schema_type == "integer":
            return 0
        if schema_type == "number":
            return 0
        if schema_type == "boolean":
            return False
        if schema_type == "array":
            return []
        if schema_type == "object":
            return {}
        return None

    def _append_schema_retry(self, tool: Tool, output: dict[str, Any]) -> None:
        schema = output.get("expected_input_schema")
        example = output.get("example")
        self._append_message(
            {
                "role": "system",
                "content": (
                    "The previous tool call was invalid. Retry the SAME tool with "
                    "arguments matching this schema.\n"
                    f"Expected input schema: {json.dumps(schema, ensure_ascii=False)}\n"
                    f"Minimal valid example arguments: {json.dumps(example, ensure_ascii=False)}"
                ),
            }
        )

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

    def _parse_tool_call_from_text(self, content: str) -> "ToolCall" | None:
        try:
            payload = json.loads(content.strip())
        except json.JSONDecodeError:
            if self.policy.red_flag_strict_json:
                return None
            try:
                payload = repair_json(content)
            except JsonRepairError:
                return None
        if not isinstance(payload, dict):
            return None
        protocol = protocol_from_payload(payload)
        if isinstance(protocol, ProtocolToolCall):
            return protocol_to_toolcall(protocol)
        return None

    def _maybe_vote_tool_call(
        self,
        tools_schemas: list[dict[str, Any]] | None,
        response: Any,
        remaining_model_calls: int,
        *,
        schema_error: bool = False,
        ambiguous: bool = False,
    ) -> tuple["ToolCall" | None, int]:
        if not self.policy.tool_vote_enabled:
            return None, remaining_model_calls
        if not (schema_error or ambiguous):
            return None, remaining_model_calls
        if not tools_schemas:
            return None, remaining_model_calls
        if remaining_model_calls <= 0:
            return None, remaining_model_calls
        before_calls = self._model_calls
        self._pending_tool_call_response = response
        elected = self._elect_tool_call(tools_schemas)
        remaining_model_calls -= self._model_calls - before_calls
        return elected, remaining_model_calls

    def _elect_tool_call(self, tools_schemas: list[dict[str, Any]]) -> "ToolCall" | None:
        from agentforge.models.base import ToolCall

        max_samples = min(
            self.policy.tool_vote_max_samples, self.policy.tool_vote_max_model_calls
        )
        pending_response = getattr(self, "_pending_tool_call_response", None)
        if pending_response is not None:
            self._pending_tool_call_response = None
        remaining_calls = max(0, self.max_model_calls - self._model_calls)
        if pending_response is None:
            max_samples = min(max_samples, remaining_calls)
        else:
            max_samples = min(max_samples, remaining_calls + 1)
        if max_samples <= 0:
            return None
        votes: dict[tuple[str, str], int] = {}
        candidates: dict[tuple[str, str], ToolCall] = {}
        for sample_idx in range(max_samples):
            if pending_response is not None:
                response = pending_response
                pending_response = None
            else:
                if remaining_calls <= 0:
                    break
                response = self.model.chat(self._messages, tools=tools_schemas)
                self._model_calls += 1
                remaining_calls -= 1
                if self.trace:
                    tool_payload = (
                        response.tool_call.model_dump()
                        if response.tool_call is not None
                        else None
                    )
                    self.trace.record_model_response(response.final_text, tool_payload)
            raw_text = response.final_text
            if raw_text and len(raw_text) > self.policy.red_flag_max_tool_call_chars:
                continue
            tool_call = response.tool_call
            if tool_call is None and raw_text:
                tool_call = self._parse_tool_call_from_text(raw_text)
            if tool_call is None:
                continue
            tool = self.registry.get(tool_call.name)
            if tool is None:
                continue
            if not self.policy.is_tool_allowed(tool_call.name):
                continue
            if not isinstance(tool_call.arguments, dict):
                continue
            try:
                validated = tool.input_schema.model_validate(tool_call.arguments)
            except ValidationError:
                continue
            arguments = validated.model_dump()
            key = (
                tool_call.name,
                json.dumps(arguments, sort_keys=True, separators=(",", ":")),
            )
            votes[key] = votes.get(key, 0) + 1
            if key not in candidates:
                candidates[key] = ToolCall(name=tool_call.name, arguments=arguments)
            counts = sorted(votes.values(), reverse=True)
            top_count = counts[0]
            second_count = counts[1] if len(counts) > 1 else 0
            if top_count - second_count >= self.policy.tool_vote_k:
                break
        if not votes:
            return None
        winning_key = max(votes, key=votes.get)
        return candidates[winning_key]

    def _code_check_loop(self, answer: str) -> str:
        current_answer = answer
        for attempt in range(self.code_check_max_iters):
            logger.info("Code check attempt %s/%s.", attempt + 1, self.code_check_max_iters)
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
            if self._pending_model_response is not None:
                response = self._pending_model_response
                self._pending_model_response = None
            else:
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
                    return current_answer
                if isinstance(protocol, ProtocolFinal):
                    current_answer = format_final(protocol.answer, protocol.checks)
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
                    "command": [sys.executable, main_file],
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

    def _detect_mcq(self, query: str) -> set[str]:
        letters: set[str] = set()
        for match in re.findall(r"(?:^|\s|\()([A-D])[\).:]", query, re.IGNORECASE):
            letters.add(match.upper())
        for match in re.findall(r"\(([A-D])\)", query, re.IGNORECASE):
            letters.add(match.upper())
        if re.search(r"\bA\s*/\s*B\s*/\s*C", query, re.IGNORECASE):
            letters.update({"A", "B", "C"})
        if "D" in query.upper() and {"A", "B", "C"}.issubset(letters):
            letters.add("D")
        return letters if len(letters) >= 3 else set()

    def _extract_mcq_letter(self, answer: str) -> str | None:
        if not answer:
            return None
        match = re.search(r"\b([A-D])\b", answer.strip(), re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r"([A-D])", answer.strip(), re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    def _coerce_mcq_answer(self, answer: str) -> str:
        letter = self._extract_mcq_letter(answer)
        if letter:
            return letter
        return "A"


def protocol_to_toolcall(protocol: ProtocolToolCall) -> "ToolCall":
    from agentforge.models.base import ToolCall

    return ToolCall(name=protocol.name, arguments=protocol.arguments)
