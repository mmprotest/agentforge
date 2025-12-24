from pathlib import Path

from agentforge.memory import MemoryStore


def test_memory_store_summarizes_bytes_output():
    store = MemoryStore()
    entry = store.add_tool_output("dummy", b"binary-data")
    assert "dummy output summary" in entry.summary


def test_memory_store_summarizes_path_output():
    store = MemoryStore()
    entry = store.add_tool_output("dummy", Path("example.txt"))
    assert "dummy output summary" in entry.summary
