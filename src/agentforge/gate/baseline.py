"""Baseline artifact resolution for gate workflows."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any


class BaselineError(RuntimeError):
    pass


def _github_api_request(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _baseline_artifact_not_found_message(artifact_name: str) -> str:
    return (
        f"Baseline artifact not found: {artifact_name}. Run the baseline workflow on main first."
    )


def _baseline_missing_file_message(expected_relpath: str) -> str:
    return (
        f"Baseline artifact missing expected file: {expected_relpath}. Ensure the baseline "
        "workflow uploads exactly this path."
    )


def _normalize_expected_relpath(expected_relpath: str) -> str:
    if not expected_relpath:
        raise BaselineError("Baseline artifact path must be a relative path.")
    normalized = expected_relpath.replace("\\", "/")
    if Path(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
        raise BaselineError(
            f"Baseline artifact path must be relative, not absolute: {expected_relpath}."
        )
    if PureWindowsPath(normalized).drive:
        raise BaselineError(
            f"Baseline artifact path must be relative, not a drive path: {expected_relpath}."
        )
    path = Path(normalized)
    if ".." in path.parts:
        raise BaselineError(
            f"Baseline artifact extraction blocked: path traversal detected for {expected_relpath}."
        )
    return path.as_posix()


def extract_single_file_from_zip(
    zip_path: Path, expected_relpath: str, out_dir: Path
) -> Path:
    normalized = _normalize_expected_relpath(expected_relpath)
    with zipfile.ZipFile(zip_path) as handle:
        try:
            info = handle.getinfo(normalized)
        except KeyError as exc:
            raise BaselineError(_baseline_missing_file_message(expected_relpath)) from exc
        if info.is_dir():
            raise BaselineError(_baseline_missing_file_message(expected_relpath))
        target_path = (out_dir / normalized).resolve()
        out_dir_resolved = out_dir.resolve()
        try:
            target_path.relative_to(out_dir_resolved)
        except ValueError as exc:
            raise BaselineError(
                "Baseline artifact extraction blocked: path traversal detected."
            ) from exc
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with handle.open(info) as source, target_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    return target_path


def _download_artifact_zip(url: str, token: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request) as response:
        destination.write_bytes(response.read())


def _find_workflow_id(api_root: str, repo: str, token: str, workflow: str) -> str:
    if workflow.endswith((".yml", ".yaml")):
        return workflow
    try:
        workflows = _github_api_request(
            f"{api_root}/repos/{repo}/actions/workflows", token
        ).get("workflows", [])
    except urllib.error.URLError as exc:
        raise BaselineError(f"GitHub API request failed: {exc}") from exc
    match = next((item for item in workflows if item.get("name") == workflow), None)
    if not match:
        raise BaselineError(f"Baseline workflow not found: {workflow}.")
    return str(match.get("id"))


def resolve_baseline_from_artifact(
    workflow: str,
    artifact_name: str,
    artifact_path: Path,
    *,
    token: str,
    repo: str,
) -> Path:
    api_root = "https://api.github.com"
    try:
        repo_data = _github_api_request(f"{api_root}/repos/{repo}", token)
    except urllib.error.URLError as exc:
        raise BaselineError(f"GitHub API request failed: {exc}") from exc
    default_branch = repo_data.get("default_branch", "main")
    workflow_id = _find_workflow_id(api_root, repo, token, workflow)
    try:
        workflow_runs = _github_api_request(
            f"{api_root}/repos/{repo}/actions/workflows/{workflow_id}/runs?"
            + urllib.parse.urlencode({"branch": default_branch, "status": "success", "per_page": 1}),
            token,
        )
    except urllib.error.URLError as exc:
        raise BaselineError(f"GitHub API request failed: {exc}") from exc
    runs = workflow_runs.get("workflow_runs", [])
    if not runs:
        raise BaselineError(_baseline_artifact_not_found_message(artifact_name))
    run_id = runs[0]["id"]
    try:
        artifacts = _github_api_request(
            f"{api_root}/repos/{repo}/actions/runs/{run_id}/artifacts", token
        ).get("artifacts", [])
    except urllib.error.URLError as exc:
        raise BaselineError(f"GitHub API request failed: {exc}") from exc
    match = next((item for item in artifacts if item.get("name") == artifact_name), None)
    if not match:
        raise BaselineError(_baseline_artifact_not_found_message(artifact_name))
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "artifact.zip"
        try:
            _download_artifact_zip(
                f"{api_root}/repos/{repo}/actions/artifacts/{match['id']}/zip", token, zip_path
            )
        except urllib.error.URLError as exc:
            raise BaselineError(f"GitHub API request failed: {exc}") from exc
        extract_dir = Path(tmp_dir) / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted = extract_single_file_from_zip(
            zip_path, artifact_path.as_posix(), extract_dir
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(extracted.read_bytes())
        return artifact_path
