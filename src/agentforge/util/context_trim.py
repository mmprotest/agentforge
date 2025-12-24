"""Context trimming utilities for message history."""

from __future__ import annotations

from typing import Any


_TURN_ROLES = {"user", "assistant", "tool"}
_TRUNCATED_HEAD_MARKER = "[TRUNCATED_HEAD]"
_TRUNCATED_TAIL_MARKER = "[TRUNCATED_TAIL]"
_ERROR_MARKERS = (
    "Traceback",
    "Exception",
    "Error:",
    "AssertionError",
    "KeyError",
    "ValueError",
    "TypeError",
)


def _message_length(message: dict[str, Any]) -> int:
    content = message.get("content")
    if content is None:
        return 0
    return len(str(content))


def _should_keep_tail(content: str) -> bool:
    return any(marker in content for marker in _ERROR_MARKERS)


def _truncate_message_content(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, str):
        return message
    if len(content) <= max_chars:
        return message
    keep_tail = _should_keep_tail(content)
    marker = _TRUNCATED_HEAD_MARKER if keep_tail else _TRUNCATED_TAIL_MARKER
    if max_chars <= len(marker):
        message["content"] = marker[:max_chars]
        return message
    keep_len = max_chars - len(marker)
    if keep_tail:
        tail = content[-keep_len:]
        message["content"] = f"{marker}{tail}"
    else:
        head = content[:keep_len]
        message["content"] = f"{head}{marker}"
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
    max_single_message_chars: int = 4000,
) -> list[dict[str, Any]]:
    """Trim messages to fit character, turn, and token approximation budgets."""
    if not messages:
        return []
    max_chars = max(1, max_chars)
    max_turns = max(1, max_turns)
    max_tokens_approx = max(1, max_tokens_approx)
    token_char_ratio = max(1, token_char_ratio)
    max_single_message_chars = max(1, max_single_message_chars)

    trimmed = [dict(message) for message in messages]
    for message in trimmed:
        _truncate_message_content(message, max_single_message_chars)
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
    while over_budget(trimmed) and removable_indices and len(trimmed) > 1:
        drop_index = removable_indices.pop(0)
        trimmed.pop(drop_index)
        removable_indices = [
            idx - 1 if idx > drop_index else idx for idx in removable_indices
        ]
        protected_indices = {
            idx - 1 if idx > drop_index else idx for idx in protected_indices
        }

    if over_budget(trimmed):
        max_chars_budget = min(max_chars, max_tokens_approx * token_char_ratio)
        total_chars = sum(_message_length(item) for item in trimmed)
        excess = total_chars - max_chars_budget
        if excess > 0:
            for message in trimmed:
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                if excess <= 0:
                    break
                if len(content) <= 1:
                    continue
                reduction = min(excess, len(content) - 1)
                new_max = len(content) - reduction
                _truncate_message_content(message, new_max)
                excess -= reduction

    return trimmed
