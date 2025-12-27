"""Safety policy configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyPolicy:
    max_steps: int = 20
    max_tool_calls: int = 6
    max_tool_creations: int = 1
    max_runtime_seconds: int = 60
    max_model_calls: int = 20
    tool_vote_enabled: bool = False
    tool_vote_k: int = 2
    tool_vote_max_samples: int = 7
    tool_vote_max_model_calls: int = 7
    red_flag_strict_json: bool = True
    red_flag_max_tool_call_chars: int = 8000
    allow_tools: list[str] | None = None

    def is_tool_allowed(self, name: str) -> bool:
        if self.allow_tools is None:
            return True
        return name in self.allow_tools
