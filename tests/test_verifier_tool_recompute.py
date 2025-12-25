from agentforge.tasks import CheckSpec, MicroTask
from agentforge.verifier import Verifier


def _tool_runner(output):
    def _runner(tool_name, args):
        return output, True

    return _runner


def _task(check: CheckSpec) -> MicroTask:
    return MicroTask(
        id="compute",
        goal="Compute",
        inputs={},
        expected_schema=None,
        tool_hint=None,
        check=check,
        status="pending",
        attempts=0,
    )


def test_tool_recompute_numeric_close():
    verifier = Verifier(_tool_runner("2.0"))
    check = CheckSpec(
        type="tool_recompute",
        params={"tool_name": "calculator", "tool_args": {"expression": "1+1"}},
    )
    assert verifier.verify("2", _task(check)).ok


def test_tool_recompute_text_normalize():
    verifier = Verifier(_tool_runner("hello world"))
    check = CheckSpec(
        type="tool_recompute",
        params={
            "tool_name": "echo",
            "tool_args": {"value": "hello world"},
            "compare": "text",
        },
    )
    assert verifier.verify("hello   world", _task(check)).ok


def test_tool_recompute_dict_subset():
    verifier = Verifier(_tool_runner({"value": 2, "extra": True}))
    check = CheckSpec(
        type="tool_recompute",
        params={
            "tool_name": "calculator",
            "tool_args": {"expression": "1+1"},
            "compare": "subset",
            "subset_direction": "expected_in_actual",
        },
    )
    assert verifier.verify({"value": 2}, _task(check)).ok
