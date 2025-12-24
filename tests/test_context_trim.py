from agentforge.util.context_trim import trim_messages


def test_trim_messages_keeps_recent_user_and_tool_summaries():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "first assistant"},
        {"role": "tool", "content": "tool one"},
        {"role": "assistant", "content": "second assistant"},
        {"role": "tool", "content": "tool two"},
        {"role": "assistant", "content": "third assistant"},
        {"role": "tool", "content": "tool three"},
        {"role": "user", "content": "latest user"},
    ]
    trimmed = trim_messages(
        messages, max_chars=80, max_turns=5, max_tokens_approx=50, token_char_ratio=4
    )
    contents = [msg["content"] for msg in trimmed]
    assert "latest user" in contents
    assert "tool two" in contents
    assert "tool three" in contents
    total_chars = sum(len(str(msg["content"])) for msg in trimmed)
    assert total_chars <= 80


def test_trim_messages_respects_token_budget():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "x" * 400},
        {"role": "assistant", "content": "y" * 400},
        {"role": "user", "content": "z" * 400},
    ]
    trimmed = trim_messages(
        messages, max_chars=2000, max_turns=10, max_tokens_approx=150, token_char_ratio=4
    )
    total_tokens = sum(
        max(1, len(str(msg["content"])) // 4)
        for msg in trimmed
        if msg.get("content")
    )
    assert total_tokens <= 150
