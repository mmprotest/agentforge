from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class EpistemicSource(Enum):
    ASSUMED = auto()
    OBSERVED = auto()
    DERIVED = auto()
    INFERRED = auto()
    VALIDATED = auto()


@dataclass
class Fact:
    content: str
    source: EpistemicSource
    provenance: Optional[str] = None
