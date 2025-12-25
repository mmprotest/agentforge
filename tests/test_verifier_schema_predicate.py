from agentforge.tasks import CheckSpec, MicroTask
from agentforge.verifier import Verifier


def _noop_tool_runner(tool_name, args):
    return {"ok": True}, True


def test_verifier_schema_and_predicate():
    verifier = Verifier(_noop_tool_runner)
    schema_task = MicroTask(
        id="schema",
        goal="Check schema",
        inputs={},
        expected_schema={"type": "object", "required": ["answer"]},
        tool_hint=None,
        check=CheckSpec(type="schema"),
        status="pending",
        attempts=0,
    )
    ok = verifier.verify({"answer": "ok"}, schema_task)
    assert ok.passed
    bad = verifier.verify({"nope": "x"}, schema_task)
    assert not bad.passed
    predicate_task = MicroTask(
        id="predicate",
        goal="Check predicate",
        inputs={},
        expected_schema=None,
        tool_hint=None,
        check=CheckSpec(type="predicate", params={"name": "non_empty"}),
        status="pending",
        attempts=0,
    )
    assert verifier.verify("hello", predicate_task).passed
    assert not verifier.verify("", predicate_task).passed
