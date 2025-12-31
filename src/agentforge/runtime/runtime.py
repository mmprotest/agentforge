"""Runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentforge.runtime.audit import AuditLogger
from agentforge.runtime.context import RunContext, new_run_context
from agentforge.runtime.observability import MetricsCollector
from agentforge.runtime.rbac import RBACConfig
from agentforge.runtime.storage import SqliteAuditStore, SqliteWorkspaceStore
from agentforge.runtime.workspaces import Workspace, ensure_workspace


@dataclass
class Runtime:
    workspace: Workspace
    audit_logger: AuditLogger
    metrics: MetricsCollector
    rbac: RBACConfig
    audit_enabled: bool = True

    def new_run_context(
        self, user_id: str | None = None, labels: dict[str, str] | None = None
    ) -> RunContext:
        return new_run_context(self.workspace.config.id, user_id=user_id, labels=labels)

    def emit_audit(
        self,
        run_context: RunContext,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if not self.audit_enabled:
            return
        self.audit_logger.emit(
            workspace_id=self.workspace.config.id,
            run_id=run_context.run_id,
            trace_id=run_context.trace_id,
            event_type=event_type,
            payload=payload,
        )

    @classmethod
    def from_workspace(
        cls,
        home_dir: str | Path,
        workspace_id: str,
        audit_enabled: bool = True,
    ) -> "Runtime":
        workspace = ensure_workspace(home_dir, workspace_id)
        db_path = workspace.path / ".agentforge.sqlite"
        audit_store = SqliteAuditStore(db_path)
        SqliteWorkspaceStore(db_path).save_workspace(
            workspace_id, workspace.config.to_dict()
        )
        audit_logger = AuditLogger(workspace.path, audit_store)
        metrics = MetricsCollector(workspace.path)
        return cls(
            workspace=workspace,
            audit_logger=audit_logger,
            metrics=metrics,
            rbac=workspace.config.rbac,
            audit_enabled=audit_enabled,
        )
