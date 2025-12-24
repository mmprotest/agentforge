"""Context compression and raw tool output storage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any
from uuid import uuid4


@dataclass
class MemoryEntry:
    tool_name: str
    handle: str
    summary: str


@dataclass
class MemoryStore:
    max_tool_output_chars: int = 4000
    keep_raw_tool_output: bool = True
    summary_lines: int = 10
    raw_outputs: dict[str, Any] = field(default_factory=dict)
    entries: list[MemoryEntry] = field(default_factory=list)
    state: dict[str, list[str]] = field(
        default_factory=lambda: {"facts": [], "intermediate_results": [], "assumptions": []}
    )

    def add_tool_output(self, tool_name: str, output: Any) -> MemoryEntry:
        handle = self._handle_for_output(output)
        summary = self._summarize_output(tool_name, output)
        if self.keep_raw_tool_output:
            self.raw_outputs[handle] = output
        entry = MemoryEntry(tool_name=tool_name, handle=handle, summary=summary)
        self.entries.append(entry)
        self.state["intermediate_results"].append(summary)
        return entry

    def _handle_for_output(self, output: Any) -> str:
        try:
            payload = json.dumps(output, sort_keys=True)
            digest = sha256(payload.encode("utf-8")).hexdigest()[:12]
            return f"tool-{digest}"
        except TypeError:
            return f"tool-{uuid4().hex[:12]}"

    def _summarize_output(self, tool_name: str, output: Any) -> str:
        text = json.dumps(output, ensure_ascii=False, indent=2)
        clipped = text[: self.max_tool_output_chars]
        lines = [line.strip() for line in clipped.splitlines() if line.strip()]
        if not lines:
            lines = ["(no output)"]
        summary = lines[: self.summary_lines]
        summary_text = "\n".join(f"- {line}" for line in summary)
        return f"{tool_name} output summary:\n{summary_text}"
