"""Safety policy configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyPolicy:
    max_tool_calls: int = 6
    max_tool_creations: int = 1
    max_runtime_seconds: int = 60
    max_model_calls: int = 20
    allow_tools: list[str] | None = None

    def is_tool_allowed(self, name: str) -> bool:
        if self.allow_tools is None:
            return True
        return name in self.allow_tools
