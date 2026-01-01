#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any


def _get_input(name: str, default: str | None = None) -> str | None:
    value = os.getenv(f"INPUT_{name.upper().replace(' ', '_')}")
    if value is None or value == "":
        return default
    return value


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_agentforge_installed() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/github/workspace"],
        check=True,
    )
    if shutil.which("agentforge") is None:
        raise SystemExit("agentforge binary not found after install")


def _resolve_workspace_path(path_value: str, workspace_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        return workspace_root / path
    if workspace_root not in path.parents and path != workspace_root:
        raise SystemExit(f"Path must be within {workspace_root}: {path}")
    return path


def _resolve_pack_name(
    eval_pack: str, workspace_root: Path, agentforge_home: Path, workspace_id: str
) -> str:
    candidate_path = Path(eval_pack)
    if not candidate_path.is_absolute():
        candidate_path = workspace_root / candidate_path
    if candidate_path.exists():
        if candidate_path.is_dir():
            pack_path = candidate_path / "pack.jsonl"
            pack_name = candidate_path.name
        else:
            pack_path = candidate_path
            pack_name = candidate_path.parent.name
        if not pack_path.exists():
            raise SystemExit("Eval pack path must include pack.jsonl")
        try:
            pack_path = pack_path.resolve()
        except FileNotFoundError:
            return eval_pack
        evals_root = (agentforge_home / "workspaces" / workspace_id / "evals").resolve()
        if evals_root in pack_path.parents and pack_path.name == "pack.jsonl":
            return pack_path.parent.name
        dest_dir = evals_root / pack_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pack_path, dest_dir / "pack.jsonl")
        return pack_name
    return eval_pack


def _write_summary(summary_path: Path, lines: list[str]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _top_failing_cases(candidate: dict[str, Any], limit: int = 10) -> list[str]:
    cases = candidate.get("cases", [])
    failing = [case.get("id", "unknown") for case in cases if not case.get("passed", True)]
    return failing[:limit]


def _build_eval_command(
    workspace_id: str,
    pack_name: str,
    report_path: Path,
    agentforge_args: list[str],
) -> list[str]:
    return [
        "agentforge",
        "--workspace",
        workspace_id,
        "eval",
        "run",
        "--pack",
        pack_name,
        "--report",
        str(report_path),
    ] + agentforge_args


def main() -> None:
    workspace_root = Path(os.getenv("GITHUB_WORKSPACE", "/github/workspace"))
    summary_path = Path(os.getenv("GITHUB_STEP_SUMMARY", "/tmp/summary.txt"))

    workspace_id = _get_input("workspace", "default") or "default"
    eval_pack_input = _get_input("eval_pack")
    if not eval_pack_input:
        raise SystemExit("Missing required input: eval_pack")
    report_out = _get_input("report_out", "agentforge_eval_report.json") or ""
    report_path = _resolve_workspace_path(report_out, workspace_root)

    baseline_report_input = _get_input("baseline_report")
    min_score = float(_get_input("min_score", "0.0") or 0.0)
    allow_regression = _parse_bool(_get_input("allow_regression", "false"), False)
    fail_on_missing_baseline = _parse_bool(
        _get_input("fail_on_missing_baseline", "true"), True
    )
    agentforge_args = shlex.split(_get_input("agentforge_args", "") or "")
    mode = _get_input("mode", "auto") or "auto"
    if mode not in {"agent", "workflow", "auto"}:
        raise SystemExit("mode must be agent, workflow, or auto")
    if not fail_on_missing_baseline:
        raise SystemExit("Unsupported: no fallbacks")
    if not baseline_report_input:
        raise SystemExit(
            "Baseline report missing. Run the baseline workflow on main first (ai-baseline.yml)."
        )

    os.environ.setdefault("AGENTFORGE_HOME", "/data")
    agentforge_home = Path(os.environ["AGENTFORGE_HOME"])
    _ensure_agentforge_installed()

    from agentforge.evals.gating import compare_reports, enforce_min_score, load_report

    pack_name = _resolve_pack_name(
        eval_pack_input, workspace_root, agentforge_home, workspace_id
    )
    command = _build_eval_command(
        workspace_id=workspace_id,
        pack_name=pack_name,
        report_path=report_path,
        agentforge_args=agentforge_args,
    )

    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    candidate = load_report(report_path)
    gate_pass = True
    baseline_score = None
    delta = None
    messages: list[str] = []

    min_pass, min_summary = enforce_min_score(candidate, min_score)
    gate_pass = gate_pass and min_pass

    baseline_path = _resolve_workspace_path(baseline_report_input, workspace_root)
    if not baseline_path.exists():
        raise SystemExit(
            "Baseline report missing. Run the baseline workflow on main first (ai-baseline.yml)."
        )
    baseline = load_report(baseline_path)
    gate_pass, compare_summary = compare_reports(
        baseline, candidate, allow_regression=allow_regression
    )
    baseline_score = compare_summary["baseline_score"]
    delta = compare_summary["delta"]
    gate_pass = gate_pass and min_pass

    candidate_score = float(candidate.get("overall_score", 0.0))
    baseline_display = "n/a" if baseline_score is None else f"{baseline_score:.4f}"
    delta_display = "n/a" if delta is None else f"{delta:.4f}"
    outcome = "PASS" if gate_pass else "FAIL"

    print(
        f"AI Regression Gate: {outcome} | candidate={candidate_score:.4f} | baseline={baseline_display} | delta={delta_display}"
    )
    for message in messages:
        print(message)

    failing_cases = _top_failing_cases(candidate)
    summary_lines = [
        "## AI Regression Gate",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Candidate score | {candidate_score:.4f} |",
        f"| Baseline score | {baseline_display} |",
        f"| Delta | {delta_display} |",
        f"| Min score | {min_score:.4f} |",
        f"| Mode | {mode} |",
        f"| Result | {outcome} |",
        "",
    ]
    if failing_cases:
        summary_lines.append("### Failing Cases (IDs)")
        summary_lines.append(", ".join(failing_cases))
        summary_lines.append("")
    if messages:
        summary_lines.append("### Messages")
        summary_lines.extend([f"- {message}" for message in messages])
        summary_lines.append("")
    _write_summary(summary_path, summary_lines)

    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
