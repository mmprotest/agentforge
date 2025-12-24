import json

from agentforge.agent import Agent
from agentforge.models.base import ModelResponse
from agentforge.models.mock import MockChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.builtins.python_sandbox import PythonSandboxTool
from agentforge.tools.registry import ToolRegistry


def test_code_check_loop_fixes_python_code(tmp_path):
    buggy = """```python
result = 1 / 0
```"""
    fixed = """```python
result = 1 + 1
```"""
    buggy_payload = json.dumps(
        {"type": "final", "answer": buggy, "confidence": 0.2, "checks": []}
    )
    fixed_payload = json.dumps(
        {"type": "final", "answer": fixed, "confidence": 0.2, "checks": []}
    )
    model = MockChatModel(
        scripted=[
            ModelResponse(final_text=buggy_payload),
            ModelResponse(final_text=fixed_payload),
        ]
    )
    registry = ToolRegistry()
    registry.register(PythonSandboxTool(str(tmp_path)))
    agent = Agent(
        model=model,
        registry=registry,
        policy=SafetyPolicy(max_model_calls=5),
        code_check=True,
        code_check_max_iters=2,
    )
    result = agent.run("write code to add numbers")
    assert "1 + 1" in result.answer


def test_code_check_allows_basic_imports(tmp_path):
    snippet = """```python
from typing import List
from collections import Counter
result = Counter(["a", "a", "b"])
```"""
    payload = json.dumps(
        {"type": "final", "answer": snippet, "confidence": 0.2, "checks": []}
    )
    model = MockChatModel(scripted=[ModelResponse(final_text=payload)])
    registry = ToolRegistry()
    registry.register(PythonSandboxTool(str(tmp_path)))
    agent = Agent(
        model=model,
        registry=registry,
        policy=SafetyPolicy(max_model_calls=2),
        code_check=True,
        code_check_max_iters=1,
    )
    result = agent.run("write python code")
    assert "Counter" in result.answer


def test_code_check_allows_non_json_reply_in_strict_mode(tmp_path):
    buggy = """```python
result = 1 / 0
```"""
    corrected = """```python
result = 1 + 1
```"""
    buggy_payload = json.dumps(
        {"type": "final", "answer": buggy, "confidence": 0.2, "checks": []}
    )
    model = MockChatModel(
        scripted=[
            ModelResponse(final_text=buggy_payload),
            ModelResponse(final_text=corrected),
        ]
    )
    registry = ToolRegistry()
    registry.register(PythonSandboxTool(str(tmp_path)))
    agent = Agent(
        model=model,
        registry=registry,
        policy=SafetyPolicy(max_model_calls=5),
        code_check=True,
        code_check_max_iters=2,
        strict_json_mode=True,
    )
    result = agent.run("write code")
    assert "1 + 1" in result.answer
