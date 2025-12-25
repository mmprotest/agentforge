from agentforge.planner import Planner, TaskGraphPlan
from agentforge.tasks import CheckSpec


def test_planner_schema_validation_rejects_invalid_graph():
    planner = Planner(max_tasks=3, max_depth=2)
    plan = TaskGraphPlan(
        version="1.0",
        objective="bad plan",
        tasks=[
            {
                "id": "a",
                "goal": "first",
                "inputs": {"depends_on": ["b"]},
                "checks": [CheckSpec(type="regex", params={"pattern": "x"})],
                "max_attempts": 1,
                "budget_hint": "low",
            },
            {
                "id": "b",
                "goal": "second",
                "inputs": {"depends_on": ["a"]},
                "checks": [CheckSpec(type="regex", params={"pattern": "x"})],
                "max_attempts": 1,
                "budget_hint": "low",
            },
        ],
        final_task_id="b",
    )
    result = planner.validate(plan)
    assert not result.ok
    assert any("cycle" in error.lower() for error in result.errors)


def test_invalid_plan_falls_back_to_default_graph(monkeypatch):
    planner = Planner()

    def _invalid_plan(_query):
        return TaskGraphPlan(
            version="1.0",
            objective="invalid",
            tasks=[
                {
                    "id": "a",
                    "goal": "bad check",
                    "inputs": {},
                    "checks": [CheckSpec(type="contains_fields", params={"required_fields": ["x"]})],
                    "max_attempts": 1,
                    "budget_hint": "low",
                }
            ],
            final_task_id="a",
        )

    monkeypatch.setattr(planner, "plan", _invalid_plan)
    plan = planner.plan("test")
    validation = planner.validate(plan)
    assert not validation.ok
    fallback = planner._default_plan("test")
    graph = planner.to_task_graph(fallback)
    assert graph.tasks[0].id == "constraints"


def test_graph_executes_tasks_in_topological_order():
    planner = Planner()
    plan = TaskGraphPlan(
        version="1.0",
        objective="ordering",
        tasks=[
            {
                "id": "a",
                "goal": "first",
                "inputs": {},
                "checks": [CheckSpec(type="regex", params={"pattern": "x"})],
                "max_attempts": 1,
                "budget_hint": "low",
            },
            {
                "id": "b",
                "goal": "second",
                "inputs": {"depends_on": ["a"]},
                "checks": [CheckSpec(type="regex", params={"pattern": "x"})],
                "max_attempts": 1,
                "budget_hint": "low",
            },
        ],
        final_task_id="b",
    )
    graph = planner.to_task_graph(plan)
    ids = [task.id for task in graph.tasks]
    assert ids.index("a") < ids.index("b")
