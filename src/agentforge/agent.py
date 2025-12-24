"""Core agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any
from uuid import uuid4

from agentforge.models.base import BaseChatModel
from agentforge.protocol import ProtocolFinal, ProtocolToolCall, format_final, parse_protocol
from agentforge.memory import MemoryStore
from agentforge.routing import suggest_tool
from agentforge.safety.policy import SafetyPolicy
from agentforge.trace import TraceRecorder
from agentforge.tools.base import Tool
from agentforge.tools.builtins.deep_think import DeepThinkTool
from agentforge.tools.registry import ToolRegistry


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
        memory: MemoryStore | None = None,
        trace: TraceRecorder | None = None,
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
        self.tools_used: list[str] = []
        self.tools_created: list[str] = []
        self._model_calls = 0

    def _internal_plan(self, query: str) -> None:
        if self.mode != "deep":
            return
        deep_tool = self.registry.get("deep_think")
        if isinstance(deep_tool, DeepThinkTool):
            result = deep_tool.run({"problem": query, "constraints": []})
            plan = result.output
        else:
            plan = {"plan": ["Use available tools"], "checks": []}
        self._messages.append(
            {
                "role": "system",
                "content": f"Internal plan (do not reveal): {plan}",
            }
        )

    def run(self, query: str) -> AgentResult:
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
            if self.verify and self._model_calls < self.max_model_calls:
                verified = self._verify_answer(query, best.answer)
                best.answer = verified.answer
                best.checks = verified.checks
                best.confidence = verified.confidence
            return best
        result = self._run_once(query)
        if self.verify and self._model_calls < self.max_model_calls:
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
        self._messages = [
            {
                "role": "system",
                "content": (
                    "You are an agent that can use tools. "
                    "Use tools when necessary. "
                    "Respond with JSON only, using either "
                    '{"type":"tool","name":"<tool>","arguments":{...}} '
                    'or {"type":"final","answer":"...","confidence":0.0,"checks":[...]} '
                    "Do not reveal chain-of-thought."
                ),
            },
        ]
        if nonce:
            self._messages.append({"role": "system", "content": f"Nonce: {nonce}"})
        self._messages.append({"role": "user", "content": query})
        self._internal_plan(query)
        if self.trace:
            self.trace.record_messages(self._messages)

        used_router = False
        for _ in range(self.policy.max_tool_calls):
            suggestion = suggest_tool(query) if not used_router else None
            if suggestion and suggestion.confidence >= 0.8:
                direct_args = self._direct_tool_args(suggestion.tool_name, query)
                if direct_args is not None:
                    tool = self.registry.get(suggestion.tool_name)
                    if tool and self.policy.is_tool_allowed(suggestion.tool_name):
                        result = tool.run(direct_args)
                        self._handle_tool_result(suggestion.tool_name, result.output, None, direct_args)
                        used_router = True
                        continue
            if suggestion and not used_router:
                self._messages.append(
                    {
                        "role": "system",
                        "content": f"Router suggestion: {suggestion.tool_name} ({suggestion.reason}).",
                    }
                )
                used_router = True
            tools = self.registry.openai_schemas()
            if self._model_calls >= self.max_model_calls:
                break
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
                protocol = parse_protocol(response.final_text)
            if response.tool_call is None and isinstance(protocol, ProtocolToolCall):
                response = response.model_copy(update={"tool_call": protocol_to_toolcall(protocol)})
            if response.final_text is not None and isinstance(protocol, ProtocolFinal):
                answer = format_final(protocol.answer, protocol.checks)
                return AgentResult(
                    answer=answer,
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    checks=protocol.checks,
                    confidence=protocol.confidence,
                    trace_path=self._finalize_trace(),
                )
            if response.final_text is not None and response.tool_call is None:
                return AgentResult(
                    answer=response.final_text,
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    trace_path=self._finalize_trace(),
                )
            if response.tool_call is None:
                return AgentResult(
                    answer="No response from model",
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    trace_path=self._finalize_trace(),
                )
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
            result = tool.run(response.tool_call.arguments)
            self._handle_tool_result(
                tool_name, result.output, response.tool_call.id, response.tool_call.arguments
            )
        return AgentResult(
            answer="Reached tool call limit",
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
    ) -> None:
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        if tool_name == "tool_maker":
            tool_created = output.get("tool") if isinstance(output, dict) else None
            if tool_created:
                self.tools_created.append(tool_created)
        entry = self.memory.add_tool_output(tool_name, output)
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id or tool_name,
            "content": json.dumps({"handle": entry.handle, "summary": entry.summary}),
        }
        self._messages.append(tool_message)
        if self.trace:
            if arguments is not None:
                self.trace.record_tool_call(tool_name, arguments)
            self.trace.record_tool_result(tool_name, entry.handle, entry.summary)

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
            "tool_calls": len(self.tools_used),
        }
        return self.trace.finalize(stats)


def protocol_to_toolcall(protocol: ProtocolToolCall) -> "ToolCall":
    from agentforge.models.base import ToolCall

    return ToolCall(name=protocol.name, arguments=protocol.arguments)
