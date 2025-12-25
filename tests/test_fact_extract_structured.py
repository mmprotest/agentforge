from agentforge.util.fact_extract import extract_facts_structured


def test_fact_extract_structured_kinds():
    structured = extract_facts_structured(
        "http_fetch",
        {"content": "See https://example.com/page and total 12kg"},
        "http_fetch output summary:\n- See https://example.com/page\n- total: 12kg",
        source="tool-123",
    )
    kinds = {item["kind"] for item in structured}
    assert "url" in kinds
    assert "number" in kinds
    assert "snippet" in kinds
    assert all(item["source"] == "tool-123" for item in structured)
