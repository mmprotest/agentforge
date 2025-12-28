"""Core agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
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
from agentforge.memory import MemoryStore
from agentforge.routing import is_code_task, should_enable_tools, suggest_tool, tool_candidates
from agentforge.safety.policy import SafetyPolicy
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
        logic_task = self._is_logic_yes_no_task(query)
        if logic_task:
            self._append_message(
                {
                    "role": "system",
                    "content": (
                        "Determine whether the conclusion follows logically. "
                        "Reason silently. Output exactly one token: Yes or No."
                    ),
                }
            )
        if nonce:
            self._append_message({"role": "system", "content": f"Nonce: {nonce}"})
        self._append_message({"role": "user", "content": query})
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

        candidates = tool_candidates(query) if tools_enabled else []
        ambiguous_tools = False
        if len(candidates) > 1:
            sorted_candidates = sorted(
                candidates, key=lambda candidate: candidate.confidence, reverse=True
            )
            if sorted_candidates[0].confidence - sorted_candidates[1].confidence <= 0.15:
                ambiguous_tools = True

        used_router = False
        strict_json_retry_remaining = 1 if self.strict_json_mode else 0
        format_contract = self._has_strict_format_contract(query)
        format_contract_retry_remaining = 1 if format_contract else 0
        tool_call_retry_used = False
        yes_no_only = bool(re.search(r"answer yes or no only", query, re.IGNORECASE))
        logic_instruction = (
            "Determine whether the conclusion follows logically. "
            "Reason silently. Output exactly one token: Yes or No."
        )
        code_check_enabled = self.code_check and is_code_task(query)
        compute_required = tools_reason == "deterministic_compute"
        compute_retry_remaining = 1 if compute_required else 0
        has_successful_compute_call = False
        remaining_steps = self.max_steps
        remaining_tool_calls = max(0, self.max_tool_calls - self._tool_calls)
        remaining_model_calls = max(0, self.max_model_calls - self._model_calls)
        logger.info(
            "Limits: steps=%s, model_calls=%s, tool_calls=%s",
            remaining_steps,
            remaining_model_calls,
            remaining_tool_calls,
        )
        while remaining_steps > 0:
            remaining_steps -= 1
            step_index = self.max_steps - remaining_steps
            logger.info(
                "Step %s/%s (remaining model_calls=%s, tool_calls=%s).",
                step_index,
                self.max_steps,
                remaining_model_calls,
                remaining_tool_calls,
            )
            suggestion = (
                suggest_tool(query) if not used_router and tools_enabled else None
            )
            if (
                suggestion
                and suggestion.confidence >= 0.8
                and remaining_tool_calls > 0
                and tools_enabled
            ):
                direct_args = self._direct_tool_args(suggestion.tool_name, query)
                if direct_args is not None:
                    tool = self.registry.get(suggestion.tool_name)
                    if tool and self.policy.is_tool_allowed(suggestion.tool_name):
                        logger.info(
                            "Router direct tool call: %s args=%s",
                            suggestion.tool_name,
                            redact(json.dumps(direct_args, ensure_ascii=False)),
                        )
                        output, ok, schema_error = self._execute_tool(tool, direct_args)
                        self._handle_tool_result(
                            suggestion.tool_name,
                            output,
                            None,
                            direct_args,
                            message_role="system",
                            explanation=(
                                "Router auto-executed a tool based on the request."
                            ),
                        )
                        used_router = True
                        if schema_error:
                            self._append_schema_retry(tool, output)
                            continue
                        remaining_tool_calls -= 1
                        self._tool_calls += 1
                        if compute_required and ok:
                            has_successful_compute_call = True
                        if ok is False:
                            continue
                        if remaining_model_calls > 0:
                            continue
                        return AgentResult(
                            answer="Reached model call limit",
                            tools_used=self.tools_used,
                            tools_created=self.tools_created,
                            trace_path=self._finalize_trace(),
                        )
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
            tools = self.registry.openai_schemas() if tools_enabled else None
            if remaining_model_calls <= 0:
                return AgentResult(
                    answer="Reached model call limit",
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    trace_path=self._finalize_trace(),
                )
            logger.info("Calling model (call #%s).", self._model_calls + 1)
            response = self.model.chat(self._messages, tools=tools)
            self._model_calls += 1
            remaining_model_calls -= 1
            logger.info(
                "Model response: final_text=%s tool_call=%s",
                bool(response.final_text),
                response.tool_call.name if response.tool_call else "none",
            )
            if self.trace:
                tool_payload = (
                    response.tool_call.model_dump()
                    if response.tool_call is not None
                    else None
                )
                self.trace.record_model_response(response.final_text, tool_payload)
            if self.strict_json_mode and (
                response.final_text is None or not response.final_text.strip()
            ):
                if strict_json_retry_remaining > 0:
                    self._append_message(
                        {
                            "role": "system",
                            "content": (
                                "You must output valid JSON only. No explanation. No tools."
                            ),
                        }
                    )
                    strict_json_retry_remaining -= 1
                    continue
                return AgentResult(
                    answer="Model returned empty output in strict JSON mode.",
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    trace_path=self._finalize_trace(),
                )
            protocol = None
            if response.final_text:
                protocol = self._parse_model_protocol(response.final_text)
                if protocol is None and self.strict_json_mode:
                    if strict_json_retry_remaining > 0:
                        self._append_message(
                            {
                                "role": "system",
                                "content": (
                                    "You must output valid JSON only. No explanation. No tools."
                                ),
                            }
                        )
                        strict_json_retry_remaining -= 1
                        continue
                    return AgentResult(
                        answer="Could not parse the model output as JSON.",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
            if response.tool_call is None and response.final_text:
                tool_call = self._parse_tool_call_from_text(response.final_text)
                if tool_call is not None:
                    response = response.model_copy(update={"tool_call": tool_call})
            protocol_tool_call = isinstance(protocol, ProtocolToolCall)
            if response.final_text is not None and isinstance(protocol, ProtocolFinal):
                answer = format_final(protocol.answer, protocol.checks)
                if self._mcq_letters:
                    answer = self._coerce_mcq_answer(answer)
                if code_check_enabled:
                    answer = self._code_check_loop(answer)
                if compute_required and not has_successful_compute_call:
                    if compute_retry_remaining > 0 and remaining_model_calls > 0:
                        self._append_message(
                            {
                                "role": "system",
                                "content": (
                                    "You must compute this using a tool. Do not guess. "
                                    "Do not answer directly."
                                ),
                            }
                        )
                        compute_retry_remaining -= 1
                        continue
                    return AgentResult(
                        answer="Computation was required but not performed using a tool.",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                if logic_task and answer.strip() not in {"Yes", "No"}:
                    if remaining_model_calls <= 0:
                        return AgentResult(
                            answer="Answer was not strictly Yes or No.",
                            tools_used=self.tools_used,
                            tools_created=self.tools_created,
                            trace_path=self._finalize_trace(),
                        )
                    self._append_message(
                        {
                            "role": "system",
                            "content": logic_instruction,
                        }
                    )
                    continue
                if yes_no_only and answer.strip() not in {"Yes", "No"}:
                    if remaining_model_calls <= 0:
                        return AgentResult(
                            answer="Answer was not strictly Yes or No.",
                            tools_used=self.tools_used,
                            tools_created=self.tools_created,
                            trace_path=self._finalize_trace(),
                        )
                    self._append_message(
                        {
                            "role": "system",
                            "content": "Answer with exactly one token: Yes or No.",
                        }
                    )
                    continue
                if format_contract and self._format_contract_violation(query, answer):
                    if format_contract_retry_remaining > 0 and remaining_model_calls > 0:
                        self._append_message(
                            {
                                "role": "system",
                                "content": (
                                    "You must return only the output in the specified format. "
                                    "No explanation. No extra text."
                                ),
                            }
                        )
                        format_contract_retry_remaining -= 1
                        continue
                    return AgentResult(
                        answer="Output violated the requested format contract.",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                logger.info("Returning final answer from protocol.")
                return AgentResult(
                    answer=answer,
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    checks=protocol.checks,
                    confidence=protocol.confidence,
                    trace_path=self._finalize_trace(),
                )
            if response.final_text is not None and response.tool_call is None:
                if protocol_tool_call:
                    if remaining_model_calls <= 0:
                        return AgentResult(
                            answer="No valid tool call from model",
                            tools_used=self.tools_used,
                            tools_created=self.tools_created,
                            trace_path=self._finalize_trace(),
                        )
                    self._append_message(
                        {
                            "role": "system",
                            "content": (
                                "The tool call format was invalid. "
                                "Respond with a valid tool call or a final answer."
                            ),
                        }
                    )
                    continue
                answer = response.final_text
                if self._mcq_letters:
                    answer = self._coerce_mcq_answer(answer)
                if code_check_enabled:
                    answer = self._code_check_loop(answer)
                if compute_required and not has_successful_compute_call:
                    if compute_retry_remaining > 0 and remaining_model_calls > 0:
                        self._append_message(
                            {
                                "role": "system",
                                "content": (
                                    "You must compute this using a tool. Do not guess. "
                                    "Do not answer directly."
                                ),
                            }
                        )
                        compute_retry_remaining -= 1
                        continue
                    return AgentResult(
                        answer="Computation was required but not performed using a tool.",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                if logic_task and answer.strip() not in {"Yes", "No"}:
                    if remaining_model_calls <= 0:
                        return AgentResult(
                            answer="Answer was not strictly Yes or No.",
                            tools_used=self.tools_used,
                            tools_created=self.tools_created,
                            trace_path=self._finalize_trace(),
                        )
                    self._append_message(
                        {
                            "role": "system",
                            "content": logic_instruction,
                        }
                    )
                    continue
                if yes_no_only and answer.strip() not in {"Yes", "No"}:
                    if remaining_model_calls <= 0:
                        return AgentResult(
                            answer="Answer was not strictly Yes or No.",
                            tools_used=self.tools_used,
                            tools_created=self.tools_created,
                            trace_path=self._finalize_trace(),
                        )
                    self._append_message(
                        {
                            "role": "system",
                            "content": "Answer with exactly one token: Yes or No.",
                        }
                    )
                    continue
                if format_contract and self._format_contract_violation(query, answer):
                    if format_contract_retry_remaining > 0 and remaining_model_calls > 0:
                        self._append_message(
                            {
                                "role": "system",
                                "content": (
                                    "You must return only the output in the specified format. "
                                    "No explanation. No extra text."
                                ),
                            }
                        )
                        format_contract_retry_remaining -= 1
                        continue
                    return AgentResult(
                        answer="Output violated the requested format contract.",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                logger.info("Returning final answer.")
                return AgentResult(
                    answer=answer,
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    trace_path=self._finalize_trace(),
                )
            if response.tool_call is None:
                if tools_enabled and ambiguous_tools:
                    elected, remaining_model_calls = self._maybe_vote_tool_call(
                        tools,
                        response,
                        remaining_model_calls,
                        ambiguous=True,
                    )
                    if elected is not None:
                        response = response.model_copy(update={"tool_call": elected})
                    else:
                        return AgentResult(
                            answer="No response from model",
                            tools_used=self.tools_used,
                            tools_created=self.tools_created,
                            trace_path=self._finalize_trace(),
                        )
                else:
                    return AgentResult(
                        answer="No response from model",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
            if not tools_enabled:
                if tool_call_retry_used:
                    return AgentResult(
                        answer="Tools are disabled and the model attempted a tool call again.",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                if remaining_model_calls <= 0:
                    return AgentResult(
                        answer="Tools are disabled and the model attempted a tool call.",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                self._append_message(
                    {
                        "role": "system",
                        "content": (
                            "You are not allowed to use tools. If the task requires "
                            "external data or computation, explicitly say so. "
                            "Otherwise, answer directly without guessing."
                        ),
                    }
                )
                tool_call_retry_used = True
                continue
            tool_name = response.tool_call.name
            if remaining_tool_calls <= 0:
                return AgentResult(
                    answer="Reached tool call limit",
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    trace_path=self._finalize_trace(),
                )
            tool = self.registry.get(tool_name)
            if tool is None or not self.policy.is_tool_allowed(tool_name):
                elected, remaining_model_calls = self._maybe_vote_tool_call(
                    tools,
                    response,
                    remaining_model_calls,
                    schema_error=True,
                    ambiguous=ambiguous_tools,
                )
                if elected is not None:
                    response = response.model_copy(update={"tool_call": elected})
                    tool_name = response.tool_call.name
                    tool = self.registry.get(tool_name)
                if tool is None:
                    return AgentResult(
                        answer=f"Requested unknown tool: {tool_name}",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
                if not self.policy.is_tool_allowed(tool_name):
                    return AgentResult(
                        answer=f"Tool not allowed: {tool_name}",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
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
                    ambiguous=ambiguous_tools,
                )
                if elected is not None:
                    response = response.model_copy(update={"tool_call": elected})
                    tool_name = response.tool_call.name
                    tool = self.registry.get(tool_name)
                    if tool is None:
                        return AgentResult(
                            answer=f"Requested unknown tool: {tool_name}",
                            tools_used=self.tools_used,
                            tools_created=self.tools_created,
                            trace_path=self._finalize_trace(),
                        )
                    if not self.policy.is_tool_allowed(tool_name):
                        return AgentResult(
                            answer=f"Tool not allowed: {tool_name}",
                            tools_used=self.tools_used,
                            tools_created=self.tools_created,
                            trace_path=self._finalize_trace(),
                        )
                    try:
                        validated_args = tool.input_schema.model_validate(
                            response.tool_call.arguments
                        ).model_dump()
                    except ValidationError as exc2:
                        output = self._tool_error_payload(tool, exc2, schema_error=True)
                        self._handle_tool_result(
                            tool_name,
                            output,
                            response.tool_call.id,
                            response.tool_call.arguments,
                        )
                        self._append_schema_retry(tool, output)
                        continue
                else:
                    output = self._tool_error_payload(tool, exc, schema_error=True)
                    self._handle_tool_result(
                        tool_name,
                        output,
                        response.tool_call.id,
                        response.tool_call.arguments,
                    )
                    self._append_schema_retry(tool, output)
                    continue
            logger.info(
                "Tool call requested: %s args=%s",
                tool_name,
                redact(json.dumps(validated_args, ensure_ascii=False)),
            )
            output, ok, schema_error = self._execute_tool(
                tool, validated_args
            )
            self._handle_tool_result(
                tool_name, output, response.tool_call.id, validated_args
            )
            if schema_error:
                self._append_schema_retry(tool, output)
                continue
            remaining_tool_calls -= 1
            self._tool_calls += 1
            if compute_required and ok:
                has_successful_compute_call = True
            if ok is False:
                continue
        return AgentResult(
            answer="Reached step limit",
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
