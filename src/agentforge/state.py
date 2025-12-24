"""Agent state for controller orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentforge.tasks import TaskGraph
from agentforge.util.progress import ProgressTracker


@dataclass
class AgentBudgets:
    model_calls: int
    tool_calls: int
    backtracks: int
    verifies: int


@dataclass
class AgentState:
    query: str
    task_graph: TaskGraph | None
    memory_state: dict[str, Any]
    last_tool_summary: str | None
    last_error: dict[str, Any] | None
    progress: ProgressTracker
    budgets: AgentBudgets
    profile: str
    task_history: list[TaskGraph] = field(default_factory=list)
    routing_prompt: str = ""
