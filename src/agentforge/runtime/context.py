"""Runtime context helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class RunContext:
    workspace_id: str
    run_id: str
    trace_id: str
    started_at: datetime
    user_id: str | None = None
    labels: dict[str, str] = field(default_factory=dict)


def new_run_context(
    workspace_id: str,
    user_id: str | None = None,
    labels: dict[str, str] | None = None,
) -> RunContext:
    return RunContext(
        workspace_id=workspace_id,
        run_id=str(uuid4()),
        trace_id=str(uuid4()),
        started_at=datetime.now(timezone.utc),
        user_id=user_id,
        labels=labels or {},
    )
