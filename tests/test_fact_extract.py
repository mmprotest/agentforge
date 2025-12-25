from agentforge.util.fact_extract import extract_facts


def test_fact_extract_urls():
    facts = extract_facts(
        "http_fetch",
        {"content": "See https://example.com/page"},
        "http_fetch output summary:\n- See https://example.com/page",
    )
    assert any("https://example.com/page" in fact for fact in facts)


def test_fact_extract_numbers_with_units():
    facts = extract_facts(
        "calculator",
        "Total: 12kg",
        "calculator output summary:\n- Total: 12kg",
    )
    assert any("12kg" in fact for fact in facts)


def test_fact_extract_key_values():
    facts = extract_facts(
        "calculator",
        {"total": 5, "note": "ok"},
        "calculator output summary:\n- total: 5",
    )
    assert "total: 5" in facts


def test_fact_extract_dedup():
    facts = extract_facts(
        "http_fetch",
        {"url": "https://example.com"},
        "http_fetch output summary:\n- https://example.com",
    )
    assert len(facts) == len(set(facts))
