"""Optional normalization helpers for benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class NormalizationOptions:
    strip_whitespace: bool = False
    uppercase_single_letter: bool = False
    extract_answer_letter: bool = False


_ANSWER_RE = re.compile(r"\banswer\s*(?:is|:)\s*([A-Za-z])\b", re.IGNORECASE)


def normalize_answer(text: str, options: NormalizationOptions) -> str:
    normalized = text
    if options.strip_whitespace:
        normalized = " ".join(normalized.split())
    if options.extract_answer_letter:
        match = _ANSWER_RE.search(normalized)
        if match:
            normalized = match.group(1)
    if options.uppercase_single_letter:
        stripped = normalized.strip()
        if len(stripped) == 1 and stripped.isalpha():
            normalized = stripped.upper()
    return normalized
