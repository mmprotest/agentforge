from __future__ import annotations

import json

from agentforge.controller import Controller
from agentforge.models.base import BaseChatModel, ModelResponse
from agentforge.state import AgentState
from agentforge.verifier import ConstraintVerifier, FormatVerifier


class ScriptedModel(BaseChatModel):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def chat(self, messages, tools=None):  # type: ignore[override]
        return ModelResponse(final_text=self._responses.pop(0))


def test_verifier_failure_not_selected() -> None:
    responses = [
        json.dumps({"action": "bad", "rationale": "nope"}),
        json.dumps({"action": "ok", "rationale": "yes"}),
    ]
    model = ScriptedModel(responses)
    controller = Controller(
        model=model,
        verifiers=[FormatVerifier(), ConstraintVerifier()],
        proposal_count=2,
        max_backtracks=0,
        max_attempts=5,
    )
    state = AgentState(task="do it", constraints=["must:ok"], artifacts={})
    answer = controller.run(state)
    assert answer == "ok"
    assert state.attempts == 1
