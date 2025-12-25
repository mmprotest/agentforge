from agentforge.agent import Agent
from agentforge.controller import Controller
from agentforge.models.mock import MockChatModel
from agentforge.policy_engine import PolicyEngine
from agentforge.profiles import get_profile
from agentforge.safety.policy import SafetyPolicy
from agentforge.state import AgentBudgets, AgentState
from agentforge.tasks import CheckSpec, MicroTask, TaskGraph
from agentforge.util.progress import ProgressTracker
from agentforge.tools.registry import ToolRegistry


def test_backtrack_restores_snapshot_state():
    agent = Agent(
        model=MockChatModel(),
        registry=ToolRegistry(),
        policy=SafetyPolicy(),
        profile="agent",
    )
    task = MicroTask(
        id="draft",
        goal="Draft response",
        inputs={},
        expected_schema=None,
        tool_hint=None,
        check=CheckSpec(type="none"),
        status="running",
        attempts=0,
    )
    task_graph = TaskGraph(tasks=[task], current_task_id="draft", history=[])
    state = AgentState(
        query="test",
        task_graph=task_graph,
        memory_state=agent.memory.state,
        last_tool_summary="initial",
        last_error=None,
        progress=ProgressTracker(),
        budgets=AgentBudgets(model_calls=1, tool_calls=1, backtracks=1, verifies=1),
        profile="agent",
        task_history=[],
        routing_prompt="",
        snapshots=[],
        last_snapshot_task_id=None,
        profile_explicit=True,
    )
    state.memory_state.update(
        {
            "facts": ["fact1"],
            "draft_answer": "draft1",
            "candidate_output": "draft1",
            "tool_error_counts": {},
            "verifier_failures": {},
            "tool_handles": [],
            "tool_handle_count": 0,
        }
    )
    controller = Controller(PolicyEngine(get_profile("agent")))
    controller._snapshot_state(state)

    entry = agent.memory.add_tool_output("calculator", {"result": 2})
    state.memory_state.update(
        {
            "facts": ["fact1", "fact2"],
            "draft_answer": "draft2",
            "candidate_output": "draft2",
            "tool_handles": [entry.handle],
            "tool_handle_count": 1,
        }
    )

    agent._apply_backtrack(state, "backtrack triggered")

    assert state.memory_state["facts"] == ["fact1"]
    assert state.memory_state["draft_answer"] == "draft1"
    assert state.memory_state["candidate_output"] == "draft1"
    assert state.memory_state["tool_handle_count"] == 0
    assert len(agent.memory.entries) == 0
