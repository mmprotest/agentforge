"""Sandbox helpers for running tests."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    success: bool
    output: str


def run_pytest(test_path: Path, timeout_seconds: int = 20) -> SandboxResult:
    """Run pytest for a specific test file in a subprocess."""
    process = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path)],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    output = (process.stdout or "") + (process.stderr or "")
    return SandboxResult(success=process.returncode == 0, output=output)
