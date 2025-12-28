"""Optional normalization helpers for benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class NormalizationOptions:
    strip_whitespace: bool = False
    uppercase_single_letter: bool = False
    extract_answer_letter: bool = False
    normalize_commas: bool = False
    strip_headers: bool = False
    allow_optional_header: bool = False


_ANSWER_RE = re.compile(r"\banswer\s*(?:is|:)\s*([A-Za-z])\b", re.IGNORECASE)
_HEADER_RE = re.compile(r"^\s*(?:final\s+)?answer\s*[:\-]\s*", re.IGNORECASE)


def normalize_answer(text: str, options: NormalizationOptions) -> str:
    normalized = text
    if options.strip_whitespace:
        normalized = " ".join(normalized.split())
    if options.strip_headers:
        normalized = _HEADER_RE.sub("", normalized).strip()
    if options.normalize_commas:
        normalized = re.sub(r"\s*,\s*", ",", normalized)
    if options.extract_answer_letter:
        match = _ANSWER_RE.search(normalized)
        if match:
            normalized = match.group(1)
    if options.uppercase_single_letter:
        stripped = normalized.strip()
        if len(stripped) == 1 and stripped.isalpha():
            normalized = stripped.upper()
    return normalized


def normalize_tabular(text: str, normalize_commas: bool = True) -> list[str]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not normalize_commas:
        return lines
    return [re.sub(r"\s*,\s*", ",", line) for line in lines]


def compare_tabular_outputs(
    expected: str,
    actual: str,
    *,
    allow_optional_header: bool = True,
    normalize_commas: bool = True,
) -> bool:
    expected_rows = normalize_tabular(expected, normalize_commas=normalize_commas)
    actual_rows = normalize_tabular(actual, normalize_commas=normalize_commas)
    if expected_rows == actual_rows:
        return True
    if not allow_optional_header:
        return False
    if len(actual_rows) == len(expected_rows) + 1 and actual_rows[1:] == expected_rows:
        return True
    if len(expected_rows) == len(actual_rows) + 1 and expected_rows[1:] == actual_rows:
        return True
    if len(expected_rows) == len(actual_rows) and expected_rows[1:] == actual_rows[1:]:
        return True
    return False
