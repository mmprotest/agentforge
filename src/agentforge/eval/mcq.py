"""Multiple-choice normalization helpers."""

from __future__ import annotations

import re
from typing import Optional


_ANSWER_PREFIX_RE = re.compile(
    r"(?i)\b(?:answer|option)\s*[:\-]?\s*\(?\s*([A-E])\s*\)?"
)
_CHOICE_TOKEN_RE = re.compile(r"(?i)\b([A-E])\b")


def normalize_mcq_answer(text: str) -> Optional[str]:
    """Return the first normalized MCQ option (A-E) or None."""
    if not text:
        return None
    normalized = text.strip()
    if not normalized:
        return None
    prefix_match = _ANSWER_PREFIX_RE.search(normalized)
    if prefix_match:
        return prefix_match.group(1).upper()
    token_match = _CHOICE_TOKEN_RE.search(normalized)
    if token_match:
        return token_match.group(1).upper()
    return None
