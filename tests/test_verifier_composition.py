from agentforge.tasks import CheckSpec, MicroTask
from agentforge.verifier import Verifier


def _noop_tool_runner(tool_name, args):
    return {"ok": True}, True


def _task(check: CheckSpec) -> MicroTask:
    return MicroTask(
        id="t1",
        goal="Test",
        inputs={},
        expected_schema=None,
        tool_hint=None,
        check=check,
        status="pending",
        attempts=0,
    )


def test_verifier_all_of_any_of():
    verifier = Verifier(_noop_tool_runner)
    all_check = CheckSpec(
        type="none",
        all_of=[
            CheckSpec(type="predicate", params={"name": "non_empty"}),
            CheckSpec(type="contains_fields", params={"required_fields": ["answer"]}),
        ],
    )
    assert verifier.verify({"answer": "ok"}, _task(all_check)).ok
    assert not verifier.verify({}, _task(all_check)).ok

    any_check = CheckSpec(
        type="none",
        any_of=[
            CheckSpec(type="regex", params={"pattern": "hi"}),
            CheckSpec(type="contains_fields", params={"required_fields": ["answer"]}),
        ],
    )
    assert verifier.verify("hi", _task(any_check)).ok
    assert verifier.verify({"answer": "ok"}, _task(any_check)).ok
    assert not verifier.verify("nope", _task(any_check)).ok


def test_verifier_contains_fields_json_protocol_tool_error_absent():
    verifier = Verifier(_noop_tool_runner)
    contains = CheckSpec(type="contains_fields", params={"required_fields": ["answer"]})
    assert verifier.verify({"answer": "ok"}, _task(contains)).ok
    assert not verifier.verify("plain", _task(contains)).ok

    protocol = CheckSpec(type="json_protocol", params={"required_keys": ["type", "answer"]})
    assert verifier.verify({"type": "final", "answer": "ok"}, _task(protocol)).ok
    assert not verifier.verify({"answer": "ok"}, _task(protocol)).ok

    tool_ok = CheckSpec(type="tool_error_absent")
    assert verifier.verify({"ok": True, "value": 1}, _task(tool_ok)).ok
    assert not verifier.verify({"ok": False, "error": "bad"}, _task(tool_ok)).ok
