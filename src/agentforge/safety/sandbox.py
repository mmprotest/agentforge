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


@dataclass
class SandboxCommandResult:
    stdout: str
    stderr: str
    exit_code: int


def run_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int = 20,
) -> SandboxCommandResult:
    """Run a command in a subprocess."""
    safe_env = sanitize_env(env)
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=cwd,
        env=safe_env,
    )
    return SandboxCommandResult(
        stdout=process.stdout or "",
        stderr=process.stderr or "",
        exit_code=process.returncode,
    )


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


def sanitize_env(env: dict[str, str]) -> dict[str, str]:
    """Return a sanitized environment for sandboxed subprocesses."""
    allowlist = {"PATH", "PYTHONPATH", "HOME", "TMPDIR", "USER"}
    passthrough = _parse_passthrough_env()
    allowlist.update(passthrough)
    filtered: dict[str, str] = {}
    for key, value in env.items():
        if _is_sensitive_key(key):
            continue
        if key in allowlist:
            filtered[key] = value
    return filtered


def _parse_passthrough_env() -> set[str]:
    raw = os.environ.get("SANDBOX_PASSTHROUGH_ENV", "")
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _is_sensitive_key(key: str) -> bool:
    upper = key.upper()
    return upper.startswith("OPENAI_") or upper.startswith("API_KEY") or upper.startswith(
        "TOKEN"
    ) or upper.startswith("SECRET")
