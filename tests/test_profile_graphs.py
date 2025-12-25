from agentforge.controller import Controller
from agentforge.policy_engine import PolicyEngine
from agentforge.profiles import get_profile


def test_controller_builds_dynamic_graphs():
    controller = Controller(PolicyEngine(get_profile("agent")))
    graph_math = controller._initial_task_graph(
        "2+2", "agent", False, code_check_enabled=False
    )
    graph_default = controller._initial_task_graph(
        "Explain the weather", "agent", False, code_check_enabled=False
    )
    assert graph_math.tasks[0].id != graph_default.tasks[0].id


def test_controller_graph_has_final_task():
    controller = Controller(PolicyEngine(get_profile("agent")))
    graph = controller._initial_task_graph(
        "convert 5 km to m", "agent", False, code_check_enabled=False
    )
    assert any(task.goal.lower().startswith("format") or task.id == "final" for task in graph.tasks)
