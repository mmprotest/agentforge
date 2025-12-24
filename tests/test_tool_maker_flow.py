from __future__ import annotations

import json

from agentforge.models.base import ModelResponse
from agentforge.models.mock import MockChatModel
from agentforge.tools.registry import ToolRegistry
from agentforge.tools.tool_maker import ToolMaker, ToolSpec, ToolMakerTool


def test_tool_maker_flow(tmp_path):
    tool_code = '''
from agentforge.tools.base import Tool, ToolResult
from .schemas import WordCountInput, WordCountOutput

class WordCountTool(Tool):
    name = "word_count"
    description = "Count words"
    input_schema = WordCountInput
    output_schema = WordCountOutput

    def run(self, data):
        payload = WordCountInput.model_validate(data)
        count = len(payload.text.split())
        return ToolResult(output=WordCountOutput(count=count).model_dump())
'''
    schemas_code = '''
from pydantic import BaseModel

class WordCountInput(BaseModel):
    text: str

class WordCountOutput(BaseModel):
    count: int
'''
    test_code = '''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from word_count.tool import WordCountTool


def test_word_count():
    tool = WordCountTool()
    result = tool.run({"text": "hello world"})
    assert result.output["count"] == 2
'''
    bundle = {
        "tool_name": "word_count",
        "tool_code": tool_code,
        "schemas_code": schemas_code,
        "test_code": test_code,
    }
    response = ModelResponse(final_text=json.dumps(bundle))
    model = MockChatModel(scripted=[response])
    registry = ToolRegistry()
    maker = ToolMaker(model, str(tmp_path))
    tool_maker_tool = ToolMakerTool(maker, registry)
    result = tool_maker_tool.run(
        {"spec": {"purpose": "count", "desired_inputs": {}, "desired_outputs": {}, "examples": []}}
    )
    assert result.output["success"] is True
    assert registry.get("word_count") is not None
    tool = registry.get("word_count")
    assert tool.run({"text": "a b c"}).output["count"] == 3
