from agentforge.controller import Controller
from agentforge.policy_engine import PolicyEngine
from agentforge.profiles import (
    build_graph_agent,
    build_graph_code,
    build_graph_math,
    build_graph_qa,
    get_profile,
)


def _is_weak_check(task) -> bool:
    check = task.check
    return (
        check.type == "predicate"
        and check.params.get("name") == "non_empty"
        and not check.all_of
        and not check.any_of
    )


def test_profile_graph_builders_are_stronger():
    graphs = [
        build_graph_agent("help me"),
        build_graph_code("write python"),
        build_graph_math("2+2"),
        build_graph_qa("what is X"),
    ]
    for graph in graphs:
        assert len(graph.tasks) <= 6
        assert any(not _is_weak_check(task) for task in graph.tasks)


def test_controller_selects_profile_graph():
    controller = Controller(PolicyEngine(get_profile("code")))
    graph_code = controller._initial_task_graph(
        "print('hi')", "code", True, code_check_enabled=True
    )
    graph_math = controller._initial_task_graph(
        "2+2", "math", True, code_check_enabled=False
    )
    assert graph_code.tasks[0].id != graph_math.tasks[0].id


def test_controller_infers_profile_when_not_explicit():
    controller = Controller(PolicyEngine(get_profile("agent")))
    graph = controller._initial_task_graph(
        "implement python function for sorting",
        "agent",
        False,
        code_check_enabled=True,
    )
    assert graph.tasks[0].id == "draft"
    assert any(not _is_weak_check(task) for task in graph.tasks)
