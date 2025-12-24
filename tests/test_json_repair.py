from agentforge.util.json_repair import repair_json


def test_json_repair_handles_code_fence_and_single_quotes():
    text = "```json\n{'type': 'tool', 'name': 'calculator', 'arguments': {'expression': '2+2',},}\n```"
    parsed = repair_json(text)
    assert parsed["name"] == "calculator"
    assert parsed["arguments"]["expression"] == "2+2"


def test_json_repair_extracts_embedded_object():
    text = "Tool call:\n{'tool': 'json_repair', 'arguments': {'text': '{\"a\":1}'}} trailing"
    parsed = repair_json(text)
    assert parsed["tool"] == "json_repair"
