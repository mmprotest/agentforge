from agentforge.controller import ActionType, Controller
from agentforge.policy_engine import PolicyEngine
from agentforge.profiles import BudgetConfig, ProfileConfig, RoutingThresholds, get_profile
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


def test_router_confidence_threshold_prevents_tool_spam():
    profile = ProfileConfig(
        name="custom",
        budgets=BudgetConfig(model_calls=5, tool_calls=5, backtracks=1, verifies=1),
        strict_json_default=False,
        verify_default=False,
        code_check_default=False,
        code_check_max_iters_default=2,
        routing_thresholds=RoutingThresholds(must_call=0.99, suggest=0.95),
    )
    policy_engine = PolicyEngine(profile)
    decision = policy_engine.route("User query: 2+2")
    assert decision is None
