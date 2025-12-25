"""Microtask graph models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CheckSpec(BaseModel):
    """Specification for verifying a microtask output."""

    type: Literal[
        "contains_fields",
        "json_protocol",
        "schema",
        "regex",
        "predicate",
        "tool_recompute",
        "code_run",
        "tool_error_absent",
        "none",
    ]
    params: dict[str, Any] = Field(default_factory=dict)
    all_of: list["CheckSpec"] | None = None
    any_of: list["CheckSpec"] | None = None


class MicroTask(BaseModel):
    """Small unit of work with verification metadata."""

    id: str
    goal: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected_schema: dict[str, Any] | None = None
    tool_hint: str | None = None
    check: CheckSpec
    status: Literal["pending", "running", "done", "failed", "blocked"]
    attempts: int
    notes: str | None = None


class TaskGraph(BaseModel):
    """Ordered microtask graph with transition history."""

    tasks: list[MicroTask]
    current_task_id: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)

    def get_task(self, task_id: str | None) -> MicroTask | None:
        if task_id is None:
            return None
        return next((task for task in self.tasks if task.id == task_id), None)

    @property
    def current_task(self) -> MicroTask | None:
        return self.get_task(self.current_task_id)

    def next_task(self) -> MicroTask | None:
        if self.current_task_id:
            current = self.get_task(self.current_task_id)
            if current and current.status == "running":
                return current
        for task in self.tasks:
            if task.status == "pending":
                task.status = "running"
                self.current_task_id = task.id
                self._record("task_started", task)
                return task
        return None

    def mark_done(self, task_id: str | None, notes: str | None = None) -> None:
        task = self.get_task(task_id)
        if not task:
            return
        task.status = "done"
        if notes:
            task.notes = notes
        self._record("task_done", task)
        self.current_task_id = None

    def mark_failed(self, task_id: str | None, notes: str | None = None) -> None:
        task = self.get_task(task_id)
        if not task:
            return
        task.status = "failed"
        if notes:
            task.notes = notes
        self._record("task_failed", task)
        self.current_task_id = None

    def mark_blocked(self, task_id: str | None, notes: str | None = None) -> None:
        task = self.get_task(task_id)
        if not task:
            return
        task.status = "blocked"
        if notes:
            task.notes = notes
        self._record("task_blocked", task)
        self.current_task_id = None

    def record_attempt(self, task_id: str | None) -> None:
        task = self.get_task(task_id)
        if not task:
            return
        task.attempts += 1
        self._record("task_attempt", task)

    def all_done(self) -> bool:
        return all(task.status == "done" for task in self.tasks)

    def previous_task_id(self) -> str | None:
        if not self.current_task_id:
            return None
        ids = [task.id for task in self.tasks]
        if self.current_task_id not in ids:
            return None
        index = ids.index(self.current_task_id)
        if index <= 0:
            return None
        return ids[index - 1]

    def _record(self, event: str, task: MicroTask) -> None:
        self.history.append(
            {
                "event": event,
                "task_id": task.id,
                "status": task.status,
                "attempts": task.attempts,
                "notes": task.notes,
            }
        )
