"""Gate eval runner and aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Any

from agentforge.evals.runner import build_eval_report
from agentforge.gate.config import GateConfig, GateConfigError


@dataclass(frozen=True)
class GatePack:
    pack_id: str
    path: Path
    mode: str


def resolve_eval_packs(config: GateConfig, root_dir: Path) -> list[GatePack]:
    packs: list[GatePack] = []
    for entry in config.eval_packs:
        pattern = (root_dir / entry.path).as_posix()
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise GateConfigError(f"eval pack path not found: {entry.path}.")
        if len(matches) == 1:
            match_path = Path(matches[0])
            if not match_path.is_file():
                raise GateConfigError(f"eval pack path is not a file: {match_path}.")
            packs.append(GatePack(pack_id=entry.id, path=match_path, mode=entry.mode))
            continue
        for match in matches:
            match_path = Path(match)
            if not match_path.is_file():
                continue
            relpath = match_path.relative_to(root_dir).as_posix()
            pack_id = f"{entry.id}:{relpath}"
            packs.append(GatePack(pack_id=pack_id, path=match_path, mode=entry.mode))
    if not packs:
        raise GateConfigError("No eval packs matched the configured paths.")
    return packs


def run_gate_packs(
    packs: list[GatePack],
    agent: Any,
    engine: Any,
) -> dict[str, Any]:
    pack_reports: list[dict[str, Any]] = []
    total_cases = 0
    total_score = 0.0
    passed_cases = 0
    failures: list[dict[str, str]] = []
    for pack in packs:
        default_mode = None if pack.mode == "auto" else pack.mode
        report = build_eval_report(
            pack.path,
            agent,
            engine,
            default_mode,
            include_payloads=False,
            include_metadata=False,
            eval_pack_id=pack.pack_id,
        )
        report["pack_path"] = pack.path.as_posix()
        pack_reports.append(report)
        cases = report.get("cases", []) or []
        total_cases += len(cases)
        passed_cases += len([case for case in cases if case.get("passed")])
        total_score += sum(float(case.get("score", 0.0)) for case in cases)
        for failure in report.get("failures", []) or []:
            failure_id = str(failure.get("id", "case"))
            reason = str(failure.get("reason", "failed"))
            failures.append({"id": f"{pack.pack_id}:{failure_id}", "reason": reason})
    overall_score = total_score / max(1, total_cases)
    return {
        "report_version": "0.1",
        "overall_score": overall_score,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failures": failures,
        "packs": pack_reports,
    }
