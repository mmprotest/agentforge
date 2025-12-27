from agentforge.memory import MemoryStore


class NonSerializable:
    def __str__(self) -> str:
        return "non-serializable-output"


def test_memory_store_summarizes_non_serializable_output():
    store = MemoryStore(max_tool_output_chars=50, summary_lines=2)
    entry = store.add_tool_output("dummy_tool", NonSerializable())
    assert "dummy_tool output summary" in entry.summary
    assert "non-serializable-output" in entry.summary
