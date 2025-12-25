"""Agent state for controller orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agentforge.tasks import TaskGraph
from agentforge.util.progress import ProgressTracker


class StateSnapshot(BaseModel):
    task_graph: TaskGraph
    memory_state_subset: dict[str, Any] = Field(default_factory=dict)
    progress_state: dict[str, Any] = Field(default_factory=dict)
    last_tool_summary: str | None = None
    last_error: dict[str, Any] | None = None
    tool_error_counts: dict[str, int] = Field(default_factory=dict)
    verifier_failures: dict[str, int] = Field(default_factory=dict)
    tool_handle_count: int = 0
    tool_handles: list[str] = Field(default_factory=list)


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
    snapshots: list[StateSnapshot] = field(default_factory=list)
    last_snapshot_task_id: str | None = None
    profile_explicit: bool = False
