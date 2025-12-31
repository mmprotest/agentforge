"""Workspace configuration and helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from agentforge.runtime.rbac import RBACConfig


DEFAULT_WORKSPACE_POLICY = {
    "tool_allowlist": None,
    "web_allowed": False,
    "python_allowed_imports": None,
    "allow_destructive_sql": False,
}


@dataclass
class WorkspaceConfig:
    id: str
    name: str
    created_at: str
    model_defaults: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_WORKSPACE_POLICY))
    rbac: RBACConfig = field(default_factory=RBACConfig)
    secrets: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rbac"] = {
            "role_tools": self.rbac.role_tools,
            "role_connectors": self.rbac.role_connectors,
            "role_workflow": self.rbac.role_workflow,
        }
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkspaceConfig":
        rbac_payload = payload.get("rbac") or {}
        rbac = RBACConfig(
            role_tools=rbac_payload.get("role_tools", {}),
            role_connectors=rbac_payload.get("role_connectors", {}),
            role_workflow=rbac_payload.get("role_workflow", {}),
        )
        return cls(
            id=payload["id"],
            name=payload.get("name", payload["id"]),
            created_at=payload.get("created_at")
            or datetime.now(timezone.utc).isoformat(),
            model_defaults=payload.get("model_defaults", {}),
            policy=payload.get("policy", dict(DEFAULT_WORKSPACE_POLICY)),
            rbac=rbac,
            secrets=payload.get("secrets", {}),
        )


@dataclass
class Workspace:
    config: WorkspaceConfig
    path: Path


def workspace_base(home_dir: str | Path) -> Path:
    return Path(home_dir).expanduser().resolve()


def workspace_path(home_dir: str | Path, workspace_id: str) -> Path:
    return workspace_base(home_dir) / "workspaces" / workspace_id


def ensure_workspace(home_dir: str | Path, workspace_id: str) -> Workspace:
    root = workspace_path(home_dir, workspace_id)
    root.mkdir(parents=True, exist_ok=True)
    for entry in ("audit", "metrics", "packs", "data", "evals"):
        (root / entry).mkdir(parents=True, exist_ok=True)
    db_path = root / ".agentforge.sqlite"
    if not db_path.exists():
        db_path.touch()
    config_path = root / "workspace.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        config = WorkspaceConfig.from_dict(payload)
    else:
        config = WorkspaceConfig(
            id=workspace_id,
            name=workspace_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        config_path.write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return Workspace(config=config, path=root)


def load_workspace(home_dir: str | Path, workspace_id: str) -> Workspace | None:
    root = workspace_path(home_dir, workspace_id)
    config_path = root / "workspace.json"
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = WorkspaceConfig.from_dict(payload)
    return Workspace(config=config, path=root)


def save_workspace(workspace: Workspace) -> None:
    config_path = workspace.path / "workspace.json"
    config_path.write_text(
        json.dumps(workspace.config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
