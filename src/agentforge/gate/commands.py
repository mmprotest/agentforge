"""Gate CLI command helpers."""

from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Any

from agentforge.evals.gating import compare, decide_pass, load_report
from agentforge.evals.junit import report_to_junit_xml, write_junit
from agentforge.evals.summary import build_summary, render_summary
from agentforge.gate.baseline import BaselineError, resolve_baseline_from_artifact
from agentforge.gate.config import GateConfigError, load_gate_config
from agentforge.gate.runner import resolve_eval_packs, run_gate_packs


GITIGNORE_ENTRIES = [
    "baseline_report.json",
    "baseline_diff.md",
    "agentforge_eval_report.json",
    "agentforge_junit.xml",
]


def _write_summary(content: str) -> None:
    summary_env = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_env:
        Path(summary_env).write_text(content, encoding="utf-8")
    else:
        print(content)


def _gitignore_update(path: Path) -> None:
    existing = []
    if path.exists():
        existing = path.read_text(encoding="utf-8").splitlines()
    entries = [entry for entry in GITIGNORE_ENTRIES if entry not in existing]
    if entries:
        content = "\n".join(existing + entries) + "\n"
        path.write_text(content, encoding="utf-8")


def init_gate(target_dir: Path, force: bool) -> None:
    gate_path = target_dir / "gate.yml"
    baseline_workflow = target_dir / ".github" / "workflows" / "ai-baseline.yml"
    gate_workflow = target_dir / ".github" / "workflows" / "ai-regression-gate.yml"
    sample_pack = target_dir / "evals" / "sample" / "pack.jsonl"

    to_check = [gate_path, baseline_workflow, gate_workflow, sample_pack]
    if not force:
        existing = [path for path in to_check if path.exists()]
        if existing:
            items = ", ".join(str(path) for path in existing)
            raise SystemExit(f"Refusing to overwrite existing files: {items}. Use --force.")

    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        "\n".join(
            [
                "version: \"v1\"",
                "eval_packs:",
                "  - id: \"sample\"",
                "    path: \"evals/sample/pack.jsonl\"",
                "baseline:",
                "  workflow: \"ai-baseline\"",
                "  artifact_name: \"ai-baseline-report\"",
                "  artifact_path: \"baseline_report.json\"",
                "gate:",
                "  min_score: 0.0",
                "  allow_regression: false",
                "outputs:",
                "  report_path: \"agentforge_eval_report.json\"",
                "  junit_path: \"agentforge_junit.xml\"",
                "  summary_max_failures: 20",
                "privacy:",
                "  redact_env_patterns:",
                "    - \"*KEY*\"",
                "    - \"*TOKEN*\"",
                "    - \"*SECRET*\"",
                "    - \"*PASSWORD*\"",
                "  never_print_payloads: true",
                "strict:",
                "  require_baseline: true",
                "  no_fallbacks: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    baseline_workflow.parent.mkdir(parents=True, exist_ok=True)
    baseline_workflow.write_text(
        "\n".join(
            [
                "name: ai-baseline",
                "on:",
                "  workflow_dispatch:",
                "jobs:",
                "  baseline:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                "      - uses: actions/checkout@v4",
                "      - uses: actions/setup-python@v5",
                "        with:",
                "          python-version: \"3.11\"",
                "      - name: Install AgentForge",
                "        run: python -m pip install \"agentforge[yaml]\"",
                "      - name: Run baseline",
                "        env:",
                "          OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}",
                "          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}",
                "          OPENAI_MODEL: ${{ secrets.MODEL || secrets.OPENAI_MODEL }}",
                "        run: agentforge gate baseline update --config gate.yml --out baseline_report.json --diff baseline_diff.md",
                "      - uses: actions/upload-artifact@v4",
                "        with:",
                "          name: ai-baseline-report",
                "          path: baseline_report.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    gate_workflow.parent.mkdir(parents=True, exist_ok=True)
    gate_workflow.write_text(
        "\n".join(
            [
                "name: ai-regression-gate",
                "on:",
                "  pull_request:",
                "jobs:",
                "  gate:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                "      - uses: actions/checkout@v4",
                "      - uses: actions/setup-python@v5",
                "        with:",
                "          python-version: \"3.11\"",
                "      - name: Install AgentForge",
                "        run: python -m pip install \"agentforge[yaml]\"",
                "      - name: Run gate",
                "        env:",
                "          OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}",
                "          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}",
                "          OPENAI_MODEL: ${{ secrets.MODEL || secrets.OPENAI_MODEL }}",
                "        run: agentforge gate check --config gate.yml",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sample_pack.parent.mkdir(parents=True, exist_ok=True)
    sample_pack.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "sample-hello",
                        "input": "Say hello.",
                        "expected_output": "hello",
                        "scoring": "contains",
                        "mode": "agent",
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    _gitignore_update(target_dir / ".gitignore")


def _render_baseline_diff(current: dict[str, Any], prev: dict[str, Any] | None) -> str:
    if prev is None:
        return "## Baseline Diff\n\nNo previous baseline provided.\n"

    def _failure_ids(report: dict[str, Any]) -> list[str]:
        return [str(item.get("id", "case")) for item in report.get("failures", []) or []]

    current_failures = _failure_ids(current)
    prev_failures = _failure_ids(prev)
    current_set = set(current_failures)
    prev_set = set(prev_failures)
    added = sorted(current_set - prev_set)
    removed = sorted(prev_set - current_set)
    unchanged = sorted(current_set & prev_set)
    top_failures = current_failures[:20]

    overall_score = float(current.get("overall_score", 0.0) or 0.0)
    total_cases = int(current.get("total_cases", 0) or 0)
    passed_cases = int(current.get("passed_cases", 0) or 0)
    return "\n".join(
        [
            "## Baseline Diff",
            "",
            f"* Overall score: {overall_score:.4f}",
            f"* Total cases: {total_cases}",
            f"* Passed cases: {passed_cases}",
            "",
            "### Failures",
            f"* Added: {', '.join(added) if added else 'None'}",
            f"* Removed: {', '.join(removed) if removed else 'None'}",
            f"* Unchanged: {', '.join(unchanged) if unchanged else 'None'}",
            "",
            "### Top failing IDs",
            ", ".join(top_failures) if top_failures else "None",
            "",
        ]
    )


def _load_prev_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"Previous baseline not found at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_update(
    config_path: Path,
    out_path: Path | None,
    diff_path: Path | None,
    prev_path: Path | None,
    *,
    settings: Any,
    runtime: Any,
    build_model: Any,
    build_registry: Any,
    build_agent: Any,
    workflow_engine_cls: Any,
) -> None:
    config = load_gate_config(config_path)
    report_path = out_path or Path(config.baseline.artifact_path)
    diff_path = diff_path or Path("baseline_diff.md")
    packs = resolve_eval_packs(config, config_path.parent)

    model = build_model(settings)
    registry = build_registry(settings, model)
    agent = build_agent(settings, model, registry, runtime=runtime)
    engine = workflow_engine_cls(model, registry, runtime=runtime)

    report = run_gate_packs(packs, agent, engine)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    prev_report = _load_prev_report(prev_path)
    diff_content = _render_baseline_diff(report, prev_report)
    diff_path.write_text(diff_content, encoding="utf-8")


def gate_check(
    config_path: Path,
    *,
    settings: Any,
    runtime: Any,
    build_model: Any,
    build_registry: Any,
    build_agent: Any,
    workflow_engine_cls: Any,
) -> None:
    config = load_gate_config(config_path)
    if config.privacy.never_print_payloads is not True:
        raise GateConfigError("privacy.never_print_payloads must be true.")
    packs = resolve_eval_packs(config, config_path.parent)

    if os.environ.get("GITHUB_ACTIONS") == "1":
        token = os.environ.get("GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPOSITORY")
        if not token or not repo:
            raise BaselineError("GITHUB_TOKEN and GITHUB_REPOSITORY are required in CI.")
        resolve_baseline_from_artifact(
            config.baseline.workflow,
            config.baseline.artifact_name,
            Path(config.baseline.artifact_path),
            token=token,
            repo=repo,
        )

    baseline_path = Path(config.baseline.artifact_path)
    if not baseline_path.exists():
        raise BaselineError(f"Baseline report missing at {baseline_path}.")

    model = build_model(settings)
    registry = build_registry(settings, model)
    agent = build_agent(settings, model, registry, runtime=runtime)
    engine = workflow_engine_cls(model, registry, runtime=runtime)

    report = run_gate_packs(packs, agent, engine)
    report_path = Path(config.outputs.report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    junit_xml = report_to_junit_xml(report)
    write_junit(Path(config.outputs.junit_path), junit_xml)

    baseline_report = load_report(baseline_path)
    compare_result = compare(baseline_report, report)
    passed = decide_pass(
        compare_result,
        min_score=config.gate.min_score,
        allow_regression=config.gate.allow_regression,
        fail_on_missing_baseline=True,
        baseline_present=True,
    )

    summary = build_summary(
        report,
        compare_result,
        passed=passed,
        max_failures=config.outputs.summary_max_failures,
    )
    if os.environ.get("GITHUB_ACTIONS") == "1":
        _write_summary(render_summary(summary))
    else:
        print(render_summary(summary))

    if not passed:
        raise SystemExit(1)
