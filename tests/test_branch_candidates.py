import json

from agentforge.agent import Agent
from agentforge.models.base import BaseChatModel, ModelResponse
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.registry import ToolRegistry


class SequenceModel(BaseChatModel):
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        if not self.responses:
            return ModelResponse(final_text=json.dumps({"type": "final", "answer": "fallback"}))
        return ModelResponse(final_text=self.responses.pop(0))


def test_branch_candidates_selects_best():
    bad = json.dumps({"type": "final", "note": "missing answer", "confidence": 0.1, "checks": []})
    good = json.dumps({"type": "final", "answer": "winner", "confidence": 0.9, "checks": []})
    model = SequenceModel([bad, good])
    agent = Agent(
        model=model,
        registry=ToolRegistry(),
        policy=SafetyPolicy(max_model_calls=3),
        profile="agent",
        branch_candidates=2,
        max_turns=2,
        strict_json_mode=True,
    )
    agent.run("hello")
    assert model.calls == 2
    assert agent._last_state is not None
    assert "winner" in str(agent._last_state.memory_state.get("candidate_output", ""))
