"""Logging helpers with secret redaction."""

from __future__ import annotations

import logging
import re
from typing import Iterable

_REDACTIONS = [
    re.compile(r"Bearer\s+[A-Za-z0-9\-_.]+"),
    re.compile(r"sk-[A-Za-z0-9]+"),
]


def redact(text: str, extra_secrets: Iterable[str] | None = None) -> str:
    """Redact known secret patterns and explicit secrets from text."""
    redacted = text
    for pattern in _REDACTIONS:
        redacted = pattern.sub("Bearer [REDACTED]", redacted)
    if extra_secrets:
        for secret in extra_secrets:
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
