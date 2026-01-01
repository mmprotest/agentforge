"""GitHub Action entrypoint for AI Regression Gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PureWindowsPath
import shlex
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any

from agentforge.cli import _load_global_config, apply_overrides
from agentforge.config import Settings
from agentforge.evals.gating import (
    ReportSchemaError,
    ReportVersionError,
    compare,
    decide_pass,
    extract_score,
    load_report,
)
from agentforge.evals.runner import run_eval_pack
from agentforge.evals.summary import build_summary, render_summary
from agentforge.factory import build_agent, build_model, build_registry
from agentforge.runtime.runtime import Runtime
from agentforge.runtime.workspaces import ensure_workspace
from agentforge.workflows.engine import WorkflowEngine


SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


class BaselineError(RuntimeError):
    pass


BASELINE_WORKFLOW_FILE = "ai-baseline.yml"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_input(name: str, default: str | None = None) -> str:
    env_name = f"INPUT_{name.upper()}"
    value = os.environ.get(env_name)
    if value is None:
        return "" if default is None else default
    return value


def _set_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
        return
    print(f"{name}={value}")


def _write_summary(content: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(content, encoding="utf-8")
    else:
        print(content)


def _redact_env_value(key: str, value: str) -> str:
    if any(marker in key.upper() for marker in SENSITIVE_ENV_MARKERS):
        return "***REDACTED***"
    return value


def _log_env(keys: list[str]) -> None:
    details = {
        key: _redact_env_value(key, os.environ.get(key, "")) for key in keys if key in os.environ
    }
    if details:
        print(f"Environment: {details}")


def _require_model_config(settings: Settings) -> None:
    missing: list[str] = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.openai_base_url:
        missing.append("OPENAI_BASE_URL")
    if not settings.openai_model:
        missing.append("OPENAI_MODEL")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing model configuration: {joined}.")


def _parse_agentforge_args(arg_string: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--mode", choices=["direct", "deep"], dest="mode")
    parser.add_argument("--allow-tool-creation", action="store_true", dest="allow_tool_creation")
    parser.add_argument("--summary-lines", type=int, dest="summary_lines")
    parser.add_argument("--max-steps", type=int, dest="max_steps")
    parser.add_argument("--max-tool-calls", type=int, dest="max_tool_calls")
    parser.add_argument("--max-model-calls", type=int, dest="max_model_calls")
    parser.add_argument("--strict-json", action="store_true", dest="strict_json")
    parser.add_argument("--code-check", action="store_true", dest="code_check")
    parser.add_argument("--code-check-max-iters", type=int, dest="code_check_max_iters")
    parser.add_argument("--max-message-chars", type=int, dest="max_message_chars")
    parser.add_argument("--max-turns", type=int, dest="max_turns")
    parser.add_argument("--tool-vote", action="store_true", dest="tool_vote")
    parser.add_argument("--tool-vote-k", type=int, dest="tool_vote_k")
    parser.add_argument("--tool-vote-max-samples", type=int, dest="tool_vote_max_samples")
    parser.add_argument("--tool-vote-max-model-calls", type=int, dest="tool_vote_max_model_calls")
    parser.add_argument("--sandbox-allowed-imports", dest="sandbox_allowed_imports")
    parser.add_argument("--no-audit", action="store_true", dest="no_audit")
    parser.add_argument("--user-role", dest="user_role", default="operator")
    args = shlex.split(arg_string)
    return parser.parse_args(args)


def _build_runtime(workspace_id: str, agentforge_args: str) -> tuple[Settings, Runtime]:
    settings = Settings()
    global_config = _load_global_config(settings.agentforge_home)
    data: dict[str, Any] = settings.model_dump()
    data.update(global_config)
    workspace = ensure_workspace(settings.agentforge_home, workspace_id)
    data.update(workspace.config.model_defaults)
    policy = workspace.config.policy or {}
    if policy.get("allow_destructive_sql") is True:
        data["allow_destructive_sql"] = True
    data["workspace_id"] = workspace_id
    data["workspace_dir"] = str(workspace.path)
    settings = Settings(**data)
    if agentforge_args:
        parsed_args = _parse_agentforge_args(agentforge_args)
        settings = apply_overrides(settings, parsed_args)
    runtime = Runtime.from_workspace(settings, workspace)
    return settings, runtime


def _github_api_request(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _baseline_missing_message() -> str:
    return (
        "Baseline report missing. Run the baseline workflow on main first "
        f"({BASELINE_WORKFLOW_FILE})."
    )


def _baseline_artifact_not_found_message(artifact_name: str) -> str:
    return (
        f"Baseline artifact not found: {artifact_name}. Run the baseline workflow on main first "
        f"({BASELINE_WORKFLOW_FILE})."
    )


def _baseline_missing_file_message(expected_relpath: str) -> str:
    return (
        f"Baseline artifact missing expected file: {expected_relpath}. Ensure ai-baseline.yml "
        "uploads exactly this path."
    )


def _normalize_expected_relpath(expected_relpath: str) -> str:
    if not expected_relpath:
        raise BaselineError("Baseline artifact path must be a relative path.")
    normalized = expected_relpath.replace("\\", "/")
    if Path(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
        raise BaselineError(
            f"Baseline artifact path must be relative, not absolute: {expected_relpath}."
        )
    if PureWindowsPath(normalized).drive:
        raise BaselineError(
            f"Baseline artifact path must be relative, not a drive path: {expected_relpath}."
        )
    path = Path(normalized)
    if ".." in path.parts:
        raise BaselineError(
            f"Baseline artifact extraction blocked: path traversal detected for {expected_relpath}."
        )
    return path.as_posix()


def extract_single_file_from_zip(
    zip_path: Path, expected_relpath: str, out_dir: Path
) -> Path:
    normalized = _normalize_expected_relpath(expected_relpath)
    with zipfile.ZipFile(zip_path) as handle:
        try:
            info = handle.getinfo(normalized)
        except KeyError as exc:
            raise BaselineError(_baseline_missing_file_message(expected_relpath)) from exc
        if info.is_dir():
            raise BaselineError(_baseline_missing_file_message(expected_relpath))
        target_path = (out_dir / normalized).resolve()
        out_dir_resolved = out_dir.resolve()
        try:
            target_path.relative_to(out_dir_resolved)
        except ValueError as exc:
            raise BaselineError(
                "Baseline artifact extraction blocked: path traversal detected."
            ) from exc
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with handle.open(info) as source, target_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    return target_path


def _download_artifact_zip(url: str, token: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request) as response:
        destination.write_bytes(response.read())


def _resolve_baseline_from_artifact(
    artifact_name: str,
    artifact_path: Path,
) -> Path:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        raise BaselineError("GITHUB_TOKEN and GITHUB_REPOSITORY are required to fetch artifacts.")
    api_root = "https://api.github.com"
    try:
        repo_data = _github_api_request(f"{api_root}/repos/{repo}", token)
    except urllib.error.URLError as exc:
        raise BaselineError(f"GitHub API request failed: {exc}") from exc
    default_branch = repo_data.get("default_branch", "main")
    try:
        workflow_runs = _github_api_request(
            f"{api_root}/repos/{repo}/actions/workflows/{BASELINE_WORKFLOW_FILE}/runs?"
            + urllib.parse.urlencode({"branch": default_branch, "status": "success", "per_page": 1}),
            token,
        )
    except urllib.error.URLError as exc:
        raise BaselineError(f"GitHub API request failed: {exc}") from exc
    runs = workflow_runs.get("workflow_runs", [])
    if not runs:
        raise BaselineError(_baseline_artifact_not_found_message(artifact_name))
    run_id = runs[0]["id"]
    try:
        artifacts = _github_api_request(
            f"{api_root}/repos/{repo}/actions/runs/{run_id}/artifacts", token
        ).get("artifacts", [])
    except urllib.error.URLError as exc:
        raise BaselineError(f"GitHub API request failed: {exc}") from exc
    match = next((item for item in artifacts if item.get("name") == artifact_name), None)
    if not match:
        raise BaselineError(_baseline_artifact_not_found_message(artifact_name))
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "artifact.zip"
        try:
            _download_artifact_zip(
                f"{api_root}/repos/{repo}/actions/artifacts/{match['id']}/zip", token, zip_path
            )
        except urllib.error.URLError as exc:
            raise BaselineError(f"GitHub API request failed: {exc}") from exc
        extract_dir = Path(tmp_dir) / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted = extract_single_file_from_zip(
            zip_path, artifact_path.as_posix(), extract_dir
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(extracted.read_bytes())
        return artifact_path


def _prepare_baseline(strategy: str, artifact_name: str, artifact_path: str) -> Path:
    if strategy != "artifact":
        raise BaselineError(f"Unsupported baseline_strategy {strategy!r}.")
    return _resolve_baseline_from_artifact(artifact_name, Path(artifact_path))


def main() -> None:
    workspace = _get_input("workspace", "default")
    eval_pack = _get_input("eval_pack")
    if not eval_pack:
        raise SystemExit("eval_pack input is required.")
    baseline_strategy = _get_input("baseline_strategy", "artifact")
    baseline_artifact_name = _get_input("baseline_artifact_name", "ai-baseline-report")
    baseline_artifact_path = _get_input("baseline_artifact_path", "baseline_report.json")
    min_score = float(_get_input("min_score", "0.0") or 0.0)
    allow_regression = _parse_bool(_get_input("allow_regression", "false"))
    fail_on_missing_baseline = _parse_bool(_get_input("fail_on_missing_baseline", "true"))
    report_out = _get_input("report_out", "agentforge_eval_report.json")
    mode = _get_input("mode", "auto")
    agentforge_args = _get_input("agentforge_args", "")

    print("AI Regression Gate starting.")
    _log_env(["OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"])

    if not fail_on_missing_baseline:
        raise SystemExit("Unsupported: no fallbacks")
    if baseline_strategy != "artifact":
        raise SystemExit(f"Unsupported baseline_strategy {baseline_strategy!r}.")

    try:
        settings, runtime = _build_runtime(workspace, agentforge_args)
        _require_model_config(settings)
        baseline_path = _prepare_baseline(
            baseline_strategy,
            baseline_artifact_name,
            baseline_artifact_path,
        )
        model = build_model(settings)
        registry = build_registry(settings, model)
        agent = build_agent(settings, model, registry, runtime=runtime)
        engine = WorkflowEngine(model, registry, runtime=runtime)
        pack_path = runtime.workspace.path / "evals" / eval_pack / "pack.jsonl"
        report_path = Path(report_out)
        if not pack_path.exists():
            raise SystemExit(f"Eval pack not found at {pack_path}")

        default_mode = None if mode == "auto" else mode
        candidate_report = run_eval_pack(pack_path, agent, engine, report_path, default_mode)

        if not baseline_path.exists():
            raise BaselineError(_baseline_missing_message())

        baseline_report = load_report(baseline_path)
        compare_result = compare(baseline_report, candidate_report)

        passed = decide_pass(
            compare_result,
            min_score=min_score,
            allow_regression=allow_regression,
            fail_on_missing_baseline=True,
            baseline_present=True,
        )
    except (BaselineError, ReportSchemaError, ReportVersionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    summary = build_summary(candidate_report, compare_result, passed=passed)
    _write_summary(render_summary(summary))

    _set_output("candidate_score", str(summary.get("candidate_score")))
    _set_output("baseline_score", str(summary.get("baseline_score")))
    _set_output("delta_score", str(summary.get("delta_score")))
    _set_output("report_path", str(report_path))
    _set_output("pass", "true" if passed else "false")

    if not passed:
        raise SystemExit("AI Regression Gate failed.")


if __name__ == "__main__":
    main()
