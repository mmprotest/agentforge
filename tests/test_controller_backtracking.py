from agentforge.agent import Agent
from agentforge.models.base import ModelResponse
from agentforge.models.mock import MockChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.registry import ToolRegistry


def test_controller_backtracks_on_repeated_failure():
    scripted = [
        ModelResponse(final_text='{"type":"final","answer":"step1","confidence":0.2,"checks":[]}'),
        ModelResponse(final_text=""),
        ModelResponse(final_text=""),
        ModelResponse(final_text='{"type":"final","answer":"step2","confidence":0.2,"checks":[]}'),
        ModelResponse(final_text='{"type":"final","answer":"ok","confidence":0.2,"checks":[]}'),
    ]
    model = MockChatModel(scripted=scripted)
    agent = Agent(
        model=model,
        registry=ToolRegistry(),
        policy=SafetyPolicy(max_model_calls=5),
        profile="qa",
        verify=False,
    )
    result = agent.run("Provide a short response")
    assert "ok" in result.answer
    assert agent._last_state is not None
    assert agent._last_state.memory_state["backtrack_count"] >= 1
