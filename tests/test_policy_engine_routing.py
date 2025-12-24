from agentforge.controller import ActionType, Controller
from agentforge.policy_engine import PolicyEngine
from agentforge.profiles import get_profile
from agentforge.state import AgentBudgets, AgentState
from agentforge.util.progress import ProgressTracker


def test_policy_engine_routes_url_deterministically():
    profile = get_profile("agent")
    policy_engine = PolicyEngine(profile)
    controller = Controller(policy_engine)
    state = AgentState(
        query="Fetch https://example.com",
        task_graph=None,
        memory_state={"code_check_enabled": False},
        last_tool_summary=None,
        last_error=None,
        progress=ProgressTracker(),
        budgets=AgentBudgets(model_calls=0, tool_calls=1, backtracks=1, verifies=1),
        profile=profile.name,
        task_history=[],
        routing_prompt="User query: Fetch https://example.com",
    )
    action = controller.decide(state)
    assert action.type == ActionType.ROUTE_TOOL
    assert action.tool_name == "http_fetch"
