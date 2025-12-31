"""Runtime storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import json
import sqlite3
from typing import Any


class AuditStore(ABC):
    @abstractmethod
    def append_event(self, event: dict[str, Any]) -> None:
        raise NotImplementedError


class MetricsStore(ABC):
    @abstractmethod
    def write_metrics(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError


class WorkspaceStore(ABC):
    @abstractmethod
    def save_workspace(self, workspace_id: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        raise NotImplementedError


class SqliteAuditStore(AuditStore):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT,
                    run_id TEXT,
                    trace_id TEXT,
                    event_type TEXT,
                    timestamp TEXT,
                    payload_json TEXT,
                    payload_hash TEXT,
                    prev_hash TEXT,
                    event_hash TEXT
                )
                """
            )
            conn.commit()

    def append_event(self, event: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit_events (
                    workspace_id, run_id, trace_id, event_type, timestamp,
                    payload_json, payload_hash, prev_hash, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("workspace_id"),
                    event.get("run_id"),
                    event.get("trace_id"),
                    event.get("event_type"),
                    event.get("timestamp"),
                    json.dumps(event.get("payload"), ensure_ascii=False),
                    event.get("payload_hash"),
                    event.get("prev_hash"),
                    event.get("event_hash"),
                ),
            )
            conn.commit()


class SqliteWorkspaceStore(WorkspaceStore):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    payload_json TEXT
                )
                """
            )
            conn.commit()

    def save_workspace(self, workspace_id: str, payload: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO workspaces (id, payload_json) VALUES (?, ?)",
                (workspace_id, json.dumps(payload, ensure_ascii=False)),
            )
            conn.commit()

    def load_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload_json FROM workspaces WHERE id = ?",
                (workspace_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0])
