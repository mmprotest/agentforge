"""Regex extraction tool."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from agentforge.tools.base import Tool, ToolResult


class RegexExtractInput(BaseModel):
    pattern: str
    text: str
    flags: list[str] = Field(default_factory=list)


class RegexExtractTool(Tool):
    name = "regex_extract"
    description = "Extract regex matches and capture groups."
    input_schema = RegexExtractInput

    def run(self, data: BaseModel | dict[str, Any]) -> ToolResult:
        payload = RegexExtractInput.model_validate(data)
        flags = _parse_flags(payload.flags)
        compiled = re.compile(payload.pattern, flags=flags)
        matches = []
        groups = []
        for match in compiled.finditer(payload.text):
            matches.append(match.group(0))
            groups.append(list(match.groups()))
        return ToolResult(output={"matches": matches, "groups": groups})


def _parse_flags(flag_names: list[str]) -> int:
    mapping = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
    }
    flags = 0
    for name in flag_names:
        flag = mapping.get(name.upper())
        if flag:
            flags |= flag
    return flags
