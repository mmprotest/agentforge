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
    last_fact_signature: str | None = None

    def update(
        self,
        fact_count: int,
        tool_count: int,
        verifier_ok: bool | None,
        issue_count: int | None,
        fact_signature: str | None = None,
    ) -> bool:
        progress = False
        if fact_count > self.last_fact_count:
            progress = True
        if fact_signature is not None and fact_signature != self.last_fact_signature:
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
        self.last_fact_signature = fact_signature
        return progress

    def reset(self) -> None:
        self.iterations_without_progress = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "last_fact_count": self.last_fact_count,
            "last_tool_handle_count": self.last_tool_handle_count,
            "last_verifier_ok": self.last_verifier_ok,
            "last_issue_count": self.last_issue_count,
            "iterations_without_progress": self.iterations_without_progress,
            "last_fact_signature": self.last_fact_signature,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProgressTracker":
        return cls(
            last_fact_count=int(payload.get("last_fact_count", 0)),
            last_tool_handle_count=int(payload.get("last_tool_handle_count", 0)),
            last_verifier_ok=payload.get("last_verifier_ok"),
            last_issue_count=payload.get("last_issue_count"),
            iterations_without_progress=int(payload.get("iterations_without_progress", 0)),
            last_fact_signature=payload.get("last_fact_signature"),
        )
