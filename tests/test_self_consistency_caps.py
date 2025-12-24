from agentforge.agent import Agent
from agentforge.models.base import ModelResponse
from agentforge.models.mock import MockChatModel
from agentforge.safety.policy import SafetyPolicy
from agentforge.tools.registry import ToolRegistry


def test_self_consistency_respects_max_model_calls():
    model = MockChatModel(
        scripted=[
            ModelResponse(final_text='{"type":"final","answer":"a","confidence":0.1,"checks":[]}'),
            ModelResponse(final_text='{"type":"final","answer":"b","confidence":0.2,"checks":[]}'),
            ModelResponse(final_text='{"type":"final","answer":"c","confidence":0.3,"checks":[]}'),
        ]
    )
    agent = Agent(
        model=model,
        registry=ToolRegistry(),
        policy=SafetyPolicy(max_model_calls=2),
        self_consistency=3,
        max_model_calls=2,
    )
    result = agent.run("test")
    assert agent._model_calls <= 2
    assert result.answer
