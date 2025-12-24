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
from agentforge.memory import MemoryEntry, MemoryStore
from agentforge.routing import is_code_task, suggest_tool
from agentforge.safety.policy import SafetyPolicy
from agentforge.trace import TraceRecorder
from agentforge.tools.base import Tool
from agentforge.tools.builtins.deep_think import DeepThinkTool
from agentforge.tools.registry import ToolRegistry
from agentforge.util.context_trim import trim_messages
from agentforge.util.json_repair import JsonRepairError, repair_json


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
                self._messages, self.max_message_chars, self.max_turns
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
        self._messages = []
        system_prompt = (
            "You are an agent that can use tools. "
            "Use tools when necessary. "
            "Respond with JSON only, using either "
            '{"type":"tool","name":"<tool>","arguments":{...}} '
            'or {"type":"final","answer":"...","confidence":0.0,"checks":[...]} '
            "Do not reveal chain-of-thought."
        )
        if self.strict_json_mode:
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

        last_tool_summary: str | None = None
        last_parse_failed: str | None = None
        format_retry_remaining = 1 if self.strict_json_mode else 0
        code_check_enabled = self.code_check and is_code_task(query)
        for _ in range(self.policy.max_tool_calls):
            routing_prompt = self._build_router_prompt(
                query, last_tool_summary, last_parse_failed
            )
            suggestion = suggest_tool(routing_prompt)
            direct_args = None
            if suggestion:
                direct_args = suggestion.suggested_args or self._direct_tool_args(
                    suggestion.tool_name, query, last_parse_failed
                )
            if suggestion and suggestion.confidence >= 0.85 and direct_args is not None:
                tool = self.registry.get(suggestion.tool_name)
                if tool and self.policy.is_tool_allowed(suggestion.tool_name):
                    output, ok = self._execute_tool(tool, direct_args)
                    entry = self._handle_tool_result(
                        suggestion.tool_name, output, None, direct_args
                    )
                    last_tool_summary = entry.summary
                    last_parse_failed = None
                    if ok is False:
                        continue
                    continue
            if suggestion and suggestion.confidence >= 0.6:
                self._append_message(
                    {
                        "role": "system",
                        "content": (
                            "Router hint: consider using "
                            f"{suggestion.tool_name} ({suggestion.reason})."
                        ),
                    }
                )
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
                protocol = self._parse_model_protocol(response.final_text)
                if protocol is None and self.strict_json_mode:
                    if format_retry_remaining > 0:
                        self._append_message(
                            {
                                "role": "system",
                                "content": (
                                    "Format error: respond with exactly one JSON object "
                                    "and nothing else."
                                ),
                            }
                        )
                        format_retry_remaining -= 1
                        last_parse_failed = response.final_text
                        continue
                    last_parse_failed = response.final_text
                    return AgentResult(
                        answer="Could not parse the model output as JSON.",
                        tools_used=self.tools_used,
                        tools_created=self.tools_created,
                        trace_path=self._finalize_trace(),
                    )
            if response.tool_call is None and isinstance(protocol, ProtocolToolCall):
                response = response.model_copy(update={"tool_call": protocol_to_toolcall(protocol)})
            if response.final_text is not None and isinstance(protocol, ProtocolFinal):
                answer = format_final(protocol.answer, protocol.checks)
                if code_check_enabled:
                    answer = self._code_check_loop(answer)
                return AgentResult(
                    answer=answer,
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                    checks=protocol.checks,
                    confidence=protocol.confidence,
                    trace_path=self._finalize_trace(),
                )
            if response.final_text is not None and response.tool_call is None:
                answer = response.final_text
                if code_check_enabled:
                    answer = self._code_check_loop(answer)
                return AgentResult(
                    answer=answer,
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
            output, ok = self._execute_tool(tool, response.tool_call.arguments)
            entry = self._handle_tool_result(
                tool_name, output, response.tool_call.id, response.tool_call.arguments
            )
            last_tool_summary = entry.summary
            last_parse_failed = None
            if ok is False:
                continue
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
    ) -> MemoryEntry:
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        if tool_name == "tool_maker":
            tool_created = output.get("tool") if isinstance(output, dict) else None
            if tool_created:
                self.tools_created.append(tool_created)
        entry = self.memory.add_tool_output(tool_name, output)
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

    def _direct_tool_args(
        self,
        tool_name: str,
        query: str,
        last_parse_failed: str | None,
    ) -> dict[str, Any] | None:
        if tool_name == "calculator":
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
                pattern_match = re.search(r"pattern\s*[:=]\s*(\S+)", query, re.IGNORECASE)
                pattern = pattern_match.group(1) if pattern_match else r".+"
            return {"pattern": pattern, "text": query}
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
