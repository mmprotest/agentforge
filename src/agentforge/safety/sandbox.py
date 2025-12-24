"""Sandbox helpers for running tests."""

from __future__ import annotations

import os
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
    repo_root = Path(__file__).resolve().parents[3]
    src_path = repo_root / "src"
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{existing_path}{os.pathsep if existing_path else ''}{src_path}"
    process = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path)],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=test_path.parent,
        env=env,
    )
    output = (process.stdout or "") + (process.stderr or "")
    return SandboxResult(success=process.returncode == 0, output=output)
