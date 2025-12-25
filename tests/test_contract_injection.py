import json

from agentforge.agent import Agent
from agentforge.models.base import BaseChatModel, ModelResponse
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.registry import ToolRegistry


class CapturingModel(BaseChatModel):
    def __init__(self) -> None:
        self.last_messages = None

    def chat(self, messages, tools):
        self.last_messages = messages
        payload = {"type": "final", "answer": "ok", "confidence": 0.5, "checks": []}
        return ModelResponse(final_text=json.dumps(payload))


def test_contract_injected_ephemeral():
    model = CapturingModel()
    agent = Agent(
        model=model,
        registry=ToolRegistry(),
        policy=SafetyPolicy(),
        profile="agent",
        strict_json_mode=None,
    )
    agent.run("Hello")
    assert model.last_messages is not None
    assert any(
        "Microtask contract:" in str(msg.get("content", ""))
        for msg in model.last_messages
    )
    contract_messages = [
        msg for msg in model.last_messages if "CONTRACT_JSON:" in str(msg.get("content", ""))
    ]
    assert contract_messages
    contract_content = str(contract_messages[-1].get("content", ""))
    json_block = contract_content.split("CONTRACT_JSON:", 1)[-1].strip()
    payload = json.loads(json_block)
    assert payload["task_id"]
    assert payload["stop_when"] == "verifier_ok"
    assert all(
        "Microtask contract:" not in str(msg.get("content", ""))
        for msg in agent._messages
    )
