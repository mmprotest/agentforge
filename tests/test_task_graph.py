from agentforge.tasks import CheckSpec, MicroTask, TaskGraph


def test_task_graph_transitions():
    tasks = [
        MicroTask(
            id="one",
            goal="First",
            inputs={},
            expected_schema=None,
            tool_hint=None,
            check=CheckSpec(type="none"),
            status="pending",
            attempts=0,
        ),
        MicroTask(
            id="two",
            goal="Second",
            inputs={},
            expected_schema=None,
            tool_hint=None,
            check=CheckSpec(type="none"),
            status="pending",
            attempts=0,
        ),
    ]
    graph = TaskGraph(tasks=tasks, current_task_id=None, history=[])
    first = graph.next_task()
    assert first is not None
    assert first.id == "one"
    graph.mark_done(first.id)
    second = graph.next_task()
    assert second is not None
    assert second.id == "two"
    graph.mark_failed(second.id, notes="failed")
    assert graph.get_task("two").status == "failed"
    assert graph.history
