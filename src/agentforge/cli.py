"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from agentforge.config import Settings
from agentforge.evals.release import ReleaseCheckError, release_check
from agentforge.evals.runner import run_eval_pack
from agentforge.factory import build_agent, build_model, build_registry
from agentforge.packs.manager import (
    build_pack,
    install_pack,
    sign_pack,
    validate_pack,
    verify_pack,
)
from agentforge.runtime.runtime import Runtime
from agentforge.runtime.workspaces import ensure_workspace, load_workspace
from agentforge.runtime.smoke import run_smoke_test
from agentforge.workflows.engine import WorkflowEngine, load_workflow_spec


SUBCOMMANDS = {
    "workflow",
    "pack",
    "eval",
    "gate",
    "release",
    "workspace",
    "metrics",
    "smoke-test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentForge CLI")
    parser.add_argument("query", type=str, help="Prompt to run")
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--mode", choices=["direct", "deep"], dest="mode")
    parser.add_argument("--allow-tool-creation", action="store_true", dest="allow_tool_creation")
    parser.add_argument("--workspace", dest="workspace")
    parser.add_argument("--verify", action="store_true", dest="verify")
    parser.add_argument("--self-consistency", type=int, dest="self_consistency", default=1)
    parser.add_argument("--max-steps", type=int, dest="max_steps")
    parser.add_argument("--max-tool-calls", type=int, dest="max_tool_calls")
    parser.add_argument("--max-model-calls", type=int, dest="max_model_calls")
    parser.add_argument("--summary-lines", type=int, dest="summary_lines")
    parser.add_argument("--strict-json", action="store_true", dest="strict_json")
    parser.add_argument("--code-check", action="store_true", dest="code_check")
    parser.add_argument("--code-check-max-iters", type=int, dest="code_check_max_iters")
    parser.add_argument("--max-message-chars", type=int, dest="max_message_chars")
    parser.add_argument("--max-turns", type=int, dest="max_turns")
    parser.add_argument("--tool-vote", action="store_true", dest="tool_vote")
    parser.add_argument("--tool-vote-k", type=int, dest="tool_vote_k")
    parser.add_argument("--tool-vote-max-samples", type=int, dest="tool_vote_max_samples")
    parser.add_argument(
        "--tool-vote-max-model-calls", type=int, dest="tool_vote_max_model_calls"
    )
    parser.add_argument(
        "--sandbox-allowed-imports", dest="sandbox_allowed_imports"
    )
    parser.add_argument("--no-audit", action="store_true", dest="no_audit")
    parser.add_argument("--user-role", dest="user_role", default="operator")
    return parser.parse_args()


def parse_subcommand(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentForge CLI")
    parser.add_argument("--workspace", dest="workspace", default=None)
    parser.add_argument("--user-role", dest="user_role", default="operator")
    parser.add_argument("--no-audit", action="store_true", dest="no_audit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    workflow = subparsers.add_parser("workflow")
    workflow_sub = workflow.add_subparsers(dest="workflow_command", required=True)
    workflow_run = workflow_sub.add_parser("run")
    workflow_run.add_argument("workflow_path")
    workflow_run.add_argument("--input", required=True)
    workflow_validate = workflow_sub.add_parser("validate")
    workflow_validate.add_argument("workflow_path")

    pack = subparsers.add_parser("pack")
    pack_sub = pack.add_subparsers(dest="pack_command", required=True)
    pack_build = pack_sub.add_parser("build")
    pack_build.add_argument("dir")
    pack_build.add_argument("--out", required=True)
    pack_sign = pack_sub.add_parser("sign")
    pack_sign.add_argument("pack")
    pack_validate = pack_sub.add_parser("validate")
    pack_validate.add_argument("pack")
    pack_verify = pack_sub.add_parser("verify")
    pack_verify.add_argument("pack")
    pack_install = pack_sub.add_parser("install")
    pack_install.add_argument("pack")
    pack_install.add_argument("--allow-unsigned", action="store_true")

    eval_cmd = subparsers.add_parser("eval")
    eval_sub = eval_cmd.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_sub.add_parser("run")
    eval_run.add_argument("--pack", required=True)
    eval_run.add_argument("--report", required=True)

    gate_cmd = subparsers.add_parser("gate")
    gate_sub = gate_cmd.add_subparsers(dest="gate_command", required=True)
    gate_run = gate_sub.add_parser("run")
    gate_run.add_argument("--pack", required=True)
    gate_run.add_argument("--baseline", required=True)
    gate_run.add_argument("--report", required=True)
    gate_run.add_argument("--min-score", type=float, default=0.0)
    gate_run.add_argument("--allow-regression", action="store_true")

    release = subparsers.add_parser("release")
    release_sub = release.add_subparsers(dest="release_command", required=True)
    release_check_cmd = release_sub.add_parser("check")
    release_check_cmd.add_argument("--baseline", required=True)
    release_check_cmd.add_argument("--candidate", required=True)
    release_check_cmd.add_argument("--min-delta", type=float, default=0.0)

    workspace = subparsers.add_parser("workspace")
    workspace_sub = workspace.add_subparsers(dest="workspace_command", required=True)
    workspace_create = workspace_sub.add_parser("create")
    workspace_create.add_argument("workspace_id")
    workspace_sub.add_parser("list")
    workspace_show = workspace_sub.add_parser("show")
    workspace_show.add_argument("workspace_id")

    metrics = subparsers.add_parser("metrics")
    metrics_sub = metrics.add_subparsers(dest="metrics_command", required=True)
    metrics_tail = metrics_sub.add_parser("tail")
    metrics_tail.add_argument("--n", type=int, default=50)
    metrics_summary = metrics_sub.add_parser("summary")
    metrics_summary.add_argument("--since", default="7d")

    smoke = subparsers.add_parser("smoke-test")

    return parser.parse_args(args)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    data: dict[str, Any] = settings.model_dump()
    if getattr(args, "base_url", None):
        data["openai_base_url"] = args.base_url
    if getattr(args, "api_key", None):
        data["openai_api_key"] = args.api_key
    if getattr(args, "model", None):
        data["openai_model"] = args.model
    if getattr(args, "mode", None):
        data["agent_mode"] = args.mode
    if getattr(args, "allow_tool_creation", False):
        data["allow_tool_creation"] = True
    if getattr(args, "summary_lines", None):
        data["summary_lines"] = args.summary_lines
    if getattr(args, "max_steps", None):
        data["max_steps"] = args.max_steps
    if getattr(args, "max_tool_calls", None):
        data["max_tool_calls"] = args.max_tool_calls
    if getattr(args, "max_model_calls", None):
        data["max_model_calls"] = args.max_model_calls
    if getattr(args, "strict_json", False):
        data["strict_json_mode"] = True
    if getattr(args, "code_check", False):
        data["code_check"] = True
    if getattr(args, "code_check_max_iters", None):
        data["code_check_max_iters"] = args.code_check_max_iters
    if getattr(args, "max_message_chars", None):
        data["max_message_chars"] = args.max_message_chars
    if getattr(args, "max_turns", None):
        data["max_turns"] = args.max_turns
    if getattr(args, "tool_vote", False):
        data["tool_vote_enabled"] = True
    if getattr(args, "tool_vote_k", None):
        data["tool_vote_k"] = args.tool_vote_k
    if getattr(args, "tool_vote_max_samples", None):
        data["tool_vote_max_samples"] = args.tool_vote_max_samples
    if getattr(args, "tool_vote_max_model_calls", None):
        data["tool_vote_max_model_calls"] = args.tool_vote_max_model_calls
    if getattr(args, "sandbox_allowed_imports", None):
        data["sandbox_allowed_imports"] = args.sandbox_allowed_imports
    return Settings(**data)


def _load_global_config(home_dir: str) -> dict[str, Any]:
    config_path = Path(home_dir).expanduser() / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _build_settings(args: argparse.Namespace, workspace_id: str) -> tuple[Settings, Runtime]:
    settings = Settings()
    global_config = _load_global_config(settings.agentforge_home)
    data = settings.model_dump()
    data.update(global_config)
    workspace = ensure_workspace(settings.agentforge_home, workspace_id)
    data.update(workspace.config.model_defaults)
    policy = workspace.config.policy or {}
    if policy.get("allow_destructive_sql") is True:
        data["allow_destructive_sql"] = True
    data["workspace_id"] = workspace_id
    data["workspace_dir"] = str(workspace.path)
    settings = Settings(**data)
    settings = apply_overrides(settings, args)
    runtime = Runtime.from_workspace(
        settings.agentforge_home,
        workspace_id,
        audit_enabled=not getattr(args, "no_audit", False),
    )
    return settings, runtime


def _load_input_payload(raw: str) -> dict[str, Any]:
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def _pack_signing_key(runtime: Runtime) -> str | None:
    import os

    ref = runtime.workspace.config.secrets.get("PACK_SIGNING_KEY")
    if isinstance(ref, str) and ref.startswith("env:"):
        return os.getenv(ref.split("env:", 1)[1])
    return ref or os.getenv("PACK_SIGNING_KEY")


def _handle_subcommand(args: argparse.Namespace) -> None:
    workspace_id = args.workspace or Settings().workspace_id
    if args.command == "workspace":
        if args.workspace_command == "create":
            workspace = ensure_workspace(Settings().agentforge_home, args.workspace_id)
            print(json.dumps(workspace.config.to_dict(), indent=2))
            return
        if args.workspace_command == "list":
            home = Path(Settings().agentforge_home).expanduser()
            root = home / "workspaces"
            if not root.exists():
                print("[]")
                return
            entries = [path.name for path in root.iterdir() if path.is_dir()]
            print(json.dumps(entries, indent=2))
            return
        if args.workspace_command == "show":
            workspace = load_workspace(Settings().agentforge_home, args.workspace_id)
            if not workspace:
                raise SystemExit("Workspace not found")
            print(json.dumps(workspace.config.to_dict(), indent=2))
            return

    settings, runtime = _build_settings(args, workspace_id)

    if args.command == "workflow":
        model = build_model(settings)
        registry = build_registry(settings, model)
        engine = WorkflowEngine(model, registry, runtime=runtime)
        if args.workflow_command == "run":
            inputs = _load_input_payload(args.input)
            spec = load_workflow_spec(args.workflow_path)
            result = engine.run(spec, inputs, runtime.new_run_context())
            print(json.dumps(result.outputs, indent=2))
            return
        if args.workflow_command == "validate":
            spec = load_workflow_spec(args.workflow_path)
            errors = engine.validate_spec(spec)
            if errors:
                raise SystemExit("\n".join(errors))
            print("ok")
            return

    if args.command == "pack":
        key = _pack_signing_key(runtime)
        if args.pack_command == "build":
            manifest = build_pack(Path(args.dir), Path(args.out))
            print(json.dumps(manifest.to_dict(), indent=2))
            return
        if args.pack_command == "sign":
            if not key:
                raise SystemExit("Missing PACK_SIGNING_KEY in workspace secrets")
            manifest = sign_pack(Path(args.pack), key)
            print(json.dumps(manifest, indent=2))
            return
        if args.pack_command == "validate":
            result = validate_pack(Path(args.pack), key=key)
            for warning in result["warnings"]:
                print(f"warning: {warning}")
            if not result["ok"]:
                raise SystemExit("\n".join(result["errors"]))
            print("ok")
            return
        if args.pack_command == "verify":
            if not key:
                raise SystemExit("Missing PACK_SIGNING_KEY in workspace secrets")
            ok = verify_pack(Path(args.pack), key)
            if not ok:
                raise SystemExit("Pack verification failed")
            print("ok")
            return
        if args.pack_command == "install":
            dest = runtime.workspace.path / "packs"
            install_pack(
                Path(args.pack),
                dest,
                allow_unsigned=args.allow_unsigned,
                key=key,
            )
            print("ok")
            return

    if args.command == "eval":
        model = build_model(settings)
        registry = build_registry(settings, model)
        agent = build_agent(settings, model, registry, runtime=runtime)
        engine = WorkflowEngine(model, registry, runtime=runtime)
        pack_path = runtime.workspace.path / "evals" / args.pack / "pack.jsonl"
        report_path = Path(args.report)
        report = run_eval_pack(pack_path, agent, engine, report_path)
        runtime.emit_audit(
            runtime.new_run_context(),
            "eval_run",
            {"pack": args.pack, "score": report.get("overall_score")},
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "gate":
        from agentforge.evals.gating import compare, decide_pass, extract_score, load_report
        from agentforge.evals.summary import build_summary, render_summary

        model = build_model(settings, use_mock=True)
        registry = build_registry(settings, model)
        agent = build_agent(settings, model, registry, runtime=runtime)
        engine = WorkflowEngine(model, registry, runtime=runtime)
        pack_path = runtime.workspace.path / "evals" / args.pack / "pack.jsonl"
        report_path = Path(args.report)
        candidate_report = run_eval_pack(pack_path, agent, engine, report_path)
        baseline_path = Path(args.baseline)
        compare_result = {
            "candidate_score": extract_score(candidate_report),
            "baseline_score": None,
            "delta": None,
        }
        baseline_present = baseline_path.exists()
        if baseline_present:
            baseline_report = load_report(baseline_path)
            compare_result = compare(baseline_report, candidate_report)
        else:
            print(
                f"Baseline report missing at {baseline_path}.",
                file=sys.stderr,
            )
        passed = decide_pass(
            compare_result,
            min_score=args.min_score,
            allow_regression=args.allow_regression,
            fail_on_missing_baseline=True,
            baseline_present=baseline_present,
        )
        summary = build_summary(candidate_report, compare_result, passed=passed)
        print(render_summary(summary))
        if not passed:
            raise SystemExit(1)
        return

    if args.command == "release":
        if args.release_command == "check":
            try:
                release_check(
                    Path(args.baseline),
                    Path(args.candidate),
                    min_delta=args.min_delta,
                )
            except ReleaseCheckError as exc:
                raise SystemExit(str(exc))
            print("ok")
            return

    if args.command == "metrics":
        metrics_dir = runtime.workspace.path / "metrics"
        if args.metrics_command == "tail":
            entries: list[str] = []
            for file_path in sorted(metrics_dir.glob("*.jsonl")):
                entries.extend(file_path.read_text(encoding="utf-8").splitlines())
            tail = entries[-args.n :]
            print("\n".join(tail))
            return
        if args.metrics_command == "summary":
            entries: list[dict[str, Any]] = []
            for file_path in sorted(metrics_dir.glob("*.jsonl")):
                for line in file_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        entries.append(json.loads(line))
            totals: dict[str, int] = {}
            for entry in entries:
                for name, value in entry.get("counters", {}).items():
                    totals[name] = totals.get(name, 0) + int(value)
            print(json.dumps({"counters": totals}, indent=2))
            return

    if args.command == "smoke-test":
        run_smoke_test(settings, runtime)
        print("ok")
        return

    raise SystemExit("Unknown command")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 2 and sys.argv[1] in SUBCOMMANDS:
        args = parse_subcommand(sys.argv[1:])
        _handle_subcommand(args)
        return
    args = parse_args()
    workspace_id = args.workspace or Settings().workspace_id
    settings, runtime = _build_settings(args, workspace_id)
    model = build_model(settings)
    registry = build_registry(settings, model)
    agent = build_agent(
        settings,
        model,
        registry,
        verify=args.verify,
        self_consistency=args.self_consistency,
        runtime=runtime,
        user_role=args.user_role,
    )
    result = agent.run(args.query)
    print("Tools used:", ", ".join(result.tools_used) or "none")
    print("Tools created:", ", ".join(result.tools_created) or "none")
    print("Verify enabled:", "yes" if args.verify else "no")
    print("Self-consistency:", args.self_consistency)
    print("Answer:\n", result.answer)


if __name__ == "__main__":
    main()
