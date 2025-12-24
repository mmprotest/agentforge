"""Core agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentforge.models.base import BaseChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.base import Tool
from agentforge.tools.builtins.deep_think import DeepThinkTool
from agentforge.tools.registry import ToolRegistry


@dataclass
class AgentResult:
    answer: str
    tools_used: list[str] = field(default_factory=list)
    tools_created: list[str] = field(default_factory=list)


class Agent:
    """Agent that can call tools and optionally create new tools."""

    def __init__(
        self,
        model: BaseChatModel,
        registry: ToolRegistry,
        policy: SafetyPolicy | None = None,
        mode: str = "direct",
    ) -> None:
        self.model = model
        self.registry = registry
        self.policy = policy or SafetyPolicy()
        self.mode = mode
        self.tools_used: list[str] = []
        self.tools_created: list[str] = []

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
        self._messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are an agent that can use tools. "
                    "Use tools when necessary. Do not reveal chain-of-thought."
                ),
            },
            {"role": "user", "content": query},
        ]
        self._internal_plan(query)

        for _ in range(self.policy.max_tool_calls):
            tools = self.registry.openai_schemas()
            response = self.model.chat(self._messages, tools=tools)
            if response.final_text is not None:
                answer = response.final_text
                return AgentResult(
                    answer=answer,
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                )
            if response.tool_call is None:
                return AgentResult(
                    answer="No response from model",
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                )
            tool_name = response.tool_call.name
            tool = self.registry.get(tool_name)
            if tool is None:
                return AgentResult(
                    answer=f"Requested unknown tool: {tool_name}",
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                )
            if not self.policy.is_tool_allowed(tool_name):
                return AgentResult(
                    answer=f"Tool not allowed: {tool_name}",
                    tools_used=self.tools_used,
                    tools_created=self.tools_created,
                )
            result = tool.run(response.tool_call.arguments)
            if tool_name not in self.tools_used:
                self.tools_used.append(tool_name)
            if tool_name == "tool_maker":
                tool_created = result.output.get("tool")
                if tool_created:
                    self.tools_created.append(tool_created)
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_name,
                    "name": tool_name,
                    "content": str(result.output),
                }
            )
        return AgentResult(
            answer="Reached tool call limit",
            tools_used=self.tools_used,
            tools_created=self.tools_created,
        )
