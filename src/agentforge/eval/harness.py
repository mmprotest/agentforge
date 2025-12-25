"""Evaluation harness with scoring and failure taxonomy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentforge.agent import Agent
from agentforge.config import Settings
from agentforge.eval.runner import build_model, build_registry
from agentforge.eval.mcq import normalize_mcq_answer
from agentforge.failures import FailureTag
from agentforge.memory import MemoryStore
from agentforge.safety.policy import SafetyPolicy
from agentforge.scoring import (
    ScoreResult,
    score_case_insensitive,
    score_exact,
    score_json_schema,
    score_numeric_tolerance,
    score_regex,
)
from agentforge.trace import TraceRecorder
from agentforge.util.logging import get_logger


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgentForge eval harness")
    parser.set_defaults(command="eval")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", required=True, help="Path to JSONL results")
    parser.add_argument(
        "--scorer",
        required=True,
        choices=["exact", "case_insensitive", "numeric_tolerance", "regex", "json_schema"],
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--eval-mode", action="store_true", dest="eval_mode")
    parser.add_argument("--self-consistency", type=int, default=1)
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--summary-lines", type=int)
    parser.add_argument("--mock", action="store_true")
    return parser


def run_eval_harness(args: argparse.Namespace) -> None:
    logger = get_logger("agentforge.eval")
    settings = Settings()
    settings.eval_mode = bool(args.eval_mode)
    if args.summary_lines is not None:
        settings.summary_lines = args.summary_lines
    if args.max_model_calls is not None:
        settings.max_model_calls = args.max_model_calls
    model = build_model(settings, use_mock=args.mock)
    registry = build_registry(settings, model)
    policy = SafetyPolicy(max_model_calls=settings.max_model_calls)

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            if not line.strip():
                continue
            payload = json.loads(line)
            sample_id = payload.get("id")
            query = payload.get("input") or payload.get("query") or ""
            expected = payload.get("expected")
            trace = TraceRecorder(
                trace_id=str(sample_id or "eval"),
                workspace_dir=settings.workspace_dir,
            )
            memory = MemoryStore(
                max_tool_output_chars=settings.max_tool_output_chars,
                keep_raw_tool_output=settings.keep_raw_tool_output,
                summary_lines=settings.summary_lines,
            )
            agent = Agent(
                model=model,
                registry=registry,
                policy=policy,
                mode=settings.agent_mode,
                verify=True,
                self_consistency=args.self_consistency,
                max_model_calls=settings.max_model_calls,
                memory=memory,
                trace=trace,
                strict_json_mode=True,
                eval_mode=True,
            )
            result = agent.run(query)
            is_mcq = _is_mcq_sample(payload, expected, query)
            score = _score_sample(args, result.answer, expected, payload, is_mcq)
            failure_tags = _collect_failure_tags(result.trace_path)
            path_type = _extract_path_type(result.trace_path)
            verifier_failures = _extract_verifier_failures(result.trace_path)
            logger.info(
                "eval.sample id=%s path=%s tools=%s verify_failures=%s score=%.2f",
                sample_id,
                path_type,
                ",".join(result.tools_used),
                ",".join(verifier_failures),
                score.score,
            )
            if is_mcq:
                normalized_pred = normalize_mcq_answer(result.answer or "")
                normalized_exp = normalize_mcq_answer("" if expected is None else str(expected))
                logger.info(
                    "eval.mcq id=%s raw_predicted=%s normalized_predicted=%s raw_expected=%s normalized_expected=%s",
                    sample_id,
                    result.answer,
                    normalized_pred,
                    expected,
                    normalized_exp,
                )
            record = {
                "id": sample_id,
                "input": query,
                "predicted": result.answer,
                "expected": expected,
                "score": score.score,
                "passed": score.passed,
                "failure_tags": failure_tags,
                "trace_path": result.trace_path,
            }
            outfile.write(json.dumps(record) + "\n")


def _score_sample(
    args: argparse.Namespace,
    predicted: str,
    expected: Any,
    payload: dict[str, Any],
    is_mcq: bool,
) -> ScoreResult:
    expected_str = "" if expected is None else str(expected)
    if is_mcq:
        normalized_pred = normalize_mcq_answer(predicted or "")
        normalized_exp = normalize_mcq_answer(expected_str)
        if not normalized_pred or not normalized_exp:
            return ScoreResult(
                score=0.0,
                passed=False,
                reason="MCQ normalization failed",
            )
        return score_exact(normalized_pred, normalized_exp)
    if args.scorer == "exact":
        return score_exact(predicted, expected_str)
    if args.scorer == "case_insensitive":
        return score_case_insensitive(predicted, expected_str)
    if args.scorer == "numeric_tolerance":
        return score_numeric_tolerance(predicted, expected_str, tolerance=args.tolerance)
    if args.scorer == "regex":
        pattern = payload.get("pattern") or expected_str
        return score_regex(predicted, str(pattern))
    if args.scorer == "json_schema":
        schema = payload.get("schema") or {}
        return score_json_schema(predicted, schema)
    return ScoreResult(score=0.0, passed=False, reason="Unknown scorer")


def _collect_failure_tags(trace_path: str | None) -> list[str]:
    if not trace_path:
        return [FailureTag.PARSE_ERROR.value]
    path = Path(trace_path)
    if not path.exists():
        return [FailureTag.PARSE_ERROR.value]
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = [
        event.get("payload", {}).get("tag")
        for event in payload.get("events", [])
        if event.get("type") == "failure"
    ]
    return [tag for tag in failures if tag]


def _is_mcq_sample(payload: dict[str, Any], expected: Any, query: str) -> bool:
    if payload.get("type") == "mcq":
        return True
    if isinstance(payload.get("choices"), list) or isinstance(payload.get("options"), list):
        return True
    normalized_expected = normalize_mcq_answer("" if expected is None else str(expected))
    if normalized_expected:
        if any(token in query for token in ["A)", "B)", "C)", "D)", "E)"]):
            return True
        if any(token in query for token in ["A.", "B.", "C.", "D.", "E."]):
            return True
        if "option" in query.lower():
            return True
    return False


def _extract_path_type(trace_path: str | None) -> str:
    payload = _load_trace(trace_path)
    if not payload:
        return "unknown"
    for event in payload.get("events", []):
        if event.get("type") == "controller_path":
            return event.get("payload", {}).get("path", "unknown")
    return "unknown"


def _extract_verifier_failures(trace_path: str | None) -> list[str]:
    payload = _load_trace(trace_path)
    if not payload:
        return []
    failures: list[str] = []
    for event in payload.get("events", []):
        if event.get("type") != "verifier":
            continue
        if event.get("payload", {}).get("ok") is False:
            for failure in event.get("payload", {}).get("failures", []):
                name = failure.get("check_name")
                if name:
                    failures.append(name)
    return failures


def _load_trace(trace_path: str | None) -> dict[str, Any] | None:
    if not trace_path:
        return None
    path = Path(trace_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
