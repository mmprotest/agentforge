from agentforge.agent import Agent
from agentforge.models.mock import MockChatModel
from agentforge.state import AgentBudgets, AgentState
from agentforge.tasks import CheckSpec, MicroTask, TaskGraph
from agentforge.tools.registry import ToolRegistry
from agentforge.util.progress import ProgressTracker
from agentforge.verifier import Verifier


def _noop_tool_runner(tool_name, args):
    return {"ok": True}, True


def test_verifier_returns_minimal_fix_payload():
    verifier = Verifier(_noop_tool_runner)
    task = MicroTask(
        id="t1",
        goal="Match regex",
        inputs={},
        expected_schema=None,
        tool_hint=None,
        check=CheckSpec(type="regex", params={"pattern": "ok"}),
        status="pending",
        attempts=0,
    )
    result = verifier.verify("nope", task)
    assert not result.passed
    assert result.failures
    assert result.failures[0].minimal_fix


def test_retry_message_contains_only_failing_constraints():
    agent = Agent(model=MockChatModel(), registry=ToolRegistry())
    task = MicroTask(
        id="final",
        goal="Finalize",
        inputs={},
        expected_schema=None,
        tool_hint=None,
        check=CheckSpec(type="regex", params={"pattern": "ok"}),
        status="pending",
        attempts=0,
    )
    graph = TaskGraph(tasks=[task], current_task_id=None, history=[])
    graph.next_task()
    state = AgentState(
        query="test",
        task_graph=graph,
        memory_state={},
        last_tool_summary=None,
        last_error={
        "failures": [
            {
                "check_name": "regex",
                "reason": "Output did not match /ok/",
                "minimal_fix": "Ensure the output matches /ok/.",
            }
        ]
        },
        progress=ProgressTracker(),
        budgets=AgentBudgets(model_calls=1, tool_calls=1, backtracks=1, verifies=1),
        profile="agent",
        task_history=[],
    )
    message = agent._build_contract_message(
        state,
        hint=None,
        retry_instruction=None,
        format_retry_message=None,
        strict_json_mode=True,
    )
    assert message is not None
    content = message["content"]
    assert "Failing checks" in content
    assert "Ensure the output matches /ok/." in content
