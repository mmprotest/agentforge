from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProblemState:
    objective: str
    constraints: List[str] = field(default_factory=list)
    known_facts: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    candidate_solution: Optional[str] = None
    confidence: Optional[float] = None
    failure_reason: Optional[str] = None
