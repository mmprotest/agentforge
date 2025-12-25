import json

from agentforge.eval import harness
from agentforge.models.base import ModelResponse
from agentforge.models.mock import MockChatModel


def test_eval_runner_writes_results_and_failure_tags(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"id":"1","input":"hi","expected":"ok"}\n', encoding="utf-8")
    output = tmp_path / "results.jsonl"
    model = MockChatModel(
        scripted=[
            ModelResponse(
                final_text='{"type":"final","answer":"ok","confidence":1,"checks":[]}'
            ),
            ModelResponse(
                final_text='{"type":"final","answer":"ok","confidence":1,"checks":[]}'
            ),
            ModelResponse(
                final_text='{"type":"final","answer":"ok","confidence":1,"checks":[]}'
            ),
        ]
    )
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(harness, "build_model", lambda settings, use_mock: model)
    args = harness.build_eval_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--scorer",
            "exact",
            "--eval-mode",
        ]
    )
    harness.run_eval_harness(args)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records
    assert "failure_tags" in records[0]
