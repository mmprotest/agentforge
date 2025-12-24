"""Trace recorder for agent runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentforge.util.logging import redact


@dataclass
class TraceRecorder:
    trace_id: str
    workspace_dir: str
    started_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append(
            {
                "type": event_type,
                "timestamp": time.time(),
                "payload": payload,
            }
        )

    def record_messages(self, messages: list[dict[str, Any]]) -> None:
        redacted = [
            {"role": msg.get("role"), "content": redact(str(msg.get("content", "")))} for msg in messages
        ]
        self.record("messages", {"messages": redacted})

    def record_model_response(self, content: str | None, tool_call: dict[str, Any] | None) -> None:
        self.record(
            "model_response",
            {
                "content": redact(content or ""),
                "tool_call": tool_call,
            },
        )

    def record_tool_result(self, tool_name: str, handle: str, summary: str) -> None:
        self.record(
            "tool_result",
            {"tool_name": tool_name, "handle": handle, "summary": summary},
        )

    def record_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self.record(
            "tool_call",
            {"tool_name": tool_name, "arguments": arguments},
        )

    def finalize(self, stats: dict[str, Any]) -> str:
        trace_dir = Path(self.workspace_dir) / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{self.trace_id}.json"
        payload = {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "ended_at": time.time(),
            "stats": stats,
            "events": self.events,
        }
        trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(trace_path)
