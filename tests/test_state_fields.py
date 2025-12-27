from __future__ import annotations

from agentforge.state import AgentState


def test_agent_state_fields_present() -> None:
    expected = {
        "task",
        "constraints",
        "artifacts",
        "history",
        "attempts",
        "branch_id",
        "current_plan",
        "verifier_results",
    }
    assert set(AgentState.model_fields.keys()) == expected
