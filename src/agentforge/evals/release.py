"""Release gating utilities."""

from __future__ import annotations

import json
from pathlib import Path


class ReleaseCheckError(RuntimeError):
    pass


def release_check(baseline_path: Path, candidate_path: Path, min_delta: float = 0.0) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_score = baseline.get("overall_score", 0.0)
    candidate_score = candidate.get("overall_score", 0.0)
    if candidate_score + min_delta < baseline_score:
        raise ReleaseCheckError(
            f"Candidate score {candidate_score} worse than baseline {baseline_score}."
        )
