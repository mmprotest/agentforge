from agentforge.agent import Agent
from agentforge.models.base import ModelResponse
from agentforge.models.mock import MockChatModel
from agentforge.tools.registry import ToolRegistry


def test_eval_mode_outputs_only_answer_no_extra_text():
    model = MockChatModel(
        scripted=[
            ModelResponse(
                final_text='{"type":"final","answer":"ok","confidence":0.2,"checks":["x"]}'
            ),
            ModelResponse(
                final_text='{"type":"final","answer":"ok","confidence":0.2,"checks":["x"]}'
            ),
            ModelResponse(
                final_text='{"type":"final","answer":"ok","confidence":0.2,"checks":["x"]}'
            ),
        ]
    )
    agent = Agent(model=model, registry=ToolRegistry(), eval_mode=True)
    result = agent.run("hello")
    assert result.answer == "ok"


def test_non_eval_mode_can_include_explanations():
    model = MockChatModel(
        scripted=[
            ModelResponse(
                final_text='{"type":"final","answer":"ok","confidence":0.2,"checks":["x"]}'
            ),
            ModelResponse(
                final_text='{"type":"final","answer":"ok","confidence":0.2,"checks":["x"]}'
            ),
            ModelResponse(
                final_text='{"type":"final","answer":"ok","confidence":0.2,"checks":["x"]}'
            ),
        ]
    )
    agent = Agent(model=model, registry=ToolRegistry())
    result = agent.run("hello")
    assert "What I did" in result.answer
