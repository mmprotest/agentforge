from __future__ import annotations

from enum import Enum, auto
import re

from agentforge.epistemic import EpistemicSource


class TaskType(Enum):
    CLOSED_BOOK = auto()
    DETERMINISTIC_COMPUTE = auto()
    DYNAMIC_INFO = auto()
    EXTRACTION = auto()
    FREEFORM = auto()


EPISTEMIC_REQUIREMENTS = {
    TaskType.CLOSED_BOOK: {EpistemicSource.ASSUMED, EpistemicSource.INFERRED},
    TaskType.DETERMINISTIC_COMPUTE: {EpistemicSource.DERIVED},
    TaskType.DYNAMIC_INFO: {EpistemicSource.OBSERVED},
    TaskType.EXTRACTION: {EpistemicSource.INFERRED, EpistemicSource.VALIDATED},
    TaskType.FREEFORM: {
        EpistemicSource.ASSUMED,
        EpistemicSource.INFERRED,
        EpistemicSource.OBSERVED,
        EpistemicSource.DERIVED,
    },
}


_EXTRACTION_RE = re.compile(r"\bextract\b|\bregex\b|\bparse\b", re.IGNORECASE)


def determine_task_type(query: str, tools_reason: str | None = None) -> TaskType:
    normalized = query.strip()
    if _EXTRACTION_RE.search(normalized):
        return TaskType.EXTRACTION
    if tools_reason == "dynamic_info":
        return TaskType.DYNAMIC_INFO
    if tools_reason == "deterministic_compute":
        return TaskType.DETERMINISTIC_COMPUTE
    if tools_reason == "short_closed_book":
        return TaskType.CLOSED_BOOK
    return TaskType.FREEFORM
