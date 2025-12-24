"""Context trimming utilities for message history."""

from __future__ import annotations

from typing import Any


_TURN_ROLES = {"user", "assistant", "tool"}


def _message_length(message: dict[str, Any]) -> int:
    content = message.get("content")
    if content is None:
        return 0
    return len(str(content))


def _truncate_tool_content(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    if message.get("role") != "tool":
        return message
    content = message.get("content")
    if not isinstance(content, str):
        return message
    if len(content) <= max_chars:
        return message
    trimmed = content[:max_chars].rstrip()
    message["content"] = f"{trimmed}…(truncated)"
    return message


def _message_tokens(message: dict[str, Any], token_char_ratio: int) -> int:
    content = message.get("content")
    if content is None:
        return 0
    length = len(str(content))
    if length == 0:
        return 0
    return max(1, length // token_char_ratio)


def trim_messages(
    messages: list[dict[str, Any]],
    max_chars: int,
    max_turns: int,
    max_tokens_approx: int,
    token_char_ratio: int,
) -> list[dict[str, Any]]:
    """Trim messages to fit character, turn, and token approximation budgets."""
    if not messages:
        return []
    max_chars = max(1, max_chars)
    max_turns = max(1, max_turns)
    max_tokens_approx = max(1, max_tokens_approx)
    token_char_ratio = max(1, token_char_ratio)

    trimmed = [dict(message) for message in messages]
    tool_max_chars = min(4000, max_chars)
    for message in trimmed:
        _truncate_tool_content(message, tool_max_chars)

    last_user_index = None
    tool_indices: list[int] = []
    for idx, message in enumerate(trimmed):
        role = message.get("role")
        if role == "user":
            last_user_index = idx
        if role == "tool":
            tool_indices.append(idx)
    protected_indices = set()
    if last_user_index is not None:
        protected_indices.add(last_user_index)
    protected_indices.update(tool_indices[-2:])

    def over_budget(items: list[dict[str, Any]]) -> bool:
        total_chars = sum(_message_length(item) for item in items)
        total_turns = sum(
            1 for item in items if item.get("role") in _TURN_ROLES
        )
        total_tokens = sum(
            _message_tokens(item, token_char_ratio) for item in items
        )
        return (
            total_chars > max_chars
            or total_turns > max_turns
            or total_tokens > max_tokens_approx
        )

    removable_indices = [
        idx
        for idx, message in enumerate(trimmed)
        if message.get("role") in _TURN_ROLES and idx not in protected_indices
    ]
    while over_budget(trimmed) and removable_indices:
        drop_index = removable_indices.pop(0)
        trimmed.pop(drop_index)
        removable_indices = [
            idx - 1 if idx > drop_index else idx for idx in removable_indices
        ]
        protected_indices = {
            idx - 1 if idx > drop_index else idx for idx in protected_indices
        }

    return trimmed
