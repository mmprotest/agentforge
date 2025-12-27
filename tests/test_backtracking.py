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


def test_backtracking_after_exhausted_proposals() -> None:
    responses = [
        json.dumps({"action": "bad", "rationale": "nope"}),
        json.dumps({"action": "worse", "rationale": "still"}),
        json.dumps({"action": "ok", "rationale": "good"}),
        json.dumps({"action": "backup", "rationale": "extra"}),
    ]
    model = ScriptedModel(responses)
    controller = Controller(
        model=model,
        verifiers=[FormatVerifier(), ConstraintVerifier()],
        proposal_count=2,
        max_backtracks=1,
        max_attempts=10,
    )
    state = AgentState(task="do it", constraints=["must:ok"], artifacts={})
    answer = controller.run(state)
    assert answer == "ok"
    assert state.history[0] == "backtrack"
    assert state.branch_id == "branch-1"
    assert state.attempts == 2
