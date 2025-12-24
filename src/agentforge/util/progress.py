"""Progress tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProgressTracker:
    last_fact_count: int = 0
    last_tool_handle_count: int = 0
    last_verifier_ok: bool | None = None
    last_issue_count: int | None = None
    iterations_without_progress: int = 0

    def update(self, fact_count: int, tool_count: int, verifier_ok: bool | None, issue_count: int | None) -> bool:
        progress = False
        if fact_count > self.last_fact_count:
            progress = True
        if tool_count > self.last_tool_handle_count:
            progress = True
        if verifier_ok is True and self.last_verifier_ok is not True:
            progress = True
        if issue_count is not None and self.last_issue_count is not None:
            if issue_count < self.last_issue_count:
                progress = True
        if progress:
            self.iterations_without_progress = 0
        else:
            self.iterations_without_progress += 1
        self.last_fact_count = fact_count
        self.last_tool_handle_count = tool_count
        self.last_verifier_ok = verifier_ok
        self.last_issue_count = issue_count
        return progress

    def reset(self) -> None:
        self.iterations_without_progress = 0
