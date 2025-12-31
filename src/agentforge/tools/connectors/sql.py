"""SQL connector tool."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from pydantic import BaseModel, Field

from agentforge.tools.base import Tool, ToolResult, ToolError


class SqlConnectorInput(BaseModel):
    engine: str = Field(default="sqlite")
    query: str
    database: str | None = None


class SqlConnectorTool(Tool):
    name = "sql_connector"
    description = "Execute SQL queries (sqlite by default)."
    input_schema = SqlConnectorInput

    def __init__(self, workspace_dir: str, allow_destructive: bool = False) -> None:
        self.workspace_dir = workspace_dir
        self.allow_destructive = allow_destructive

    def run(self, data: BaseModel | dict[str, Any]) -> ToolResult:
        payload = SqlConnectorInput.model_validate(data)
        if not self.allow_destructive and not self._is_select_only(payload.query):
            raise ToolError("Destructive queries are disabled by workspace policy.")
        if payload.engine == "sqlite":
            return self._run_sqlite(payload)
        if payload.engine == "postgres":
            return self._run_postgres(payload)
        raise ToolError("Unsupported SQL engine")

    def _is_select_only(self, query: str) -> bool:
        return bool(re.match(r"\s*select\b", query, re.IGNORECASE))

    def _run_sqlite(self, payload: SqlConnectorInput) -> ToolResult:
        db_path = payload.database or ":memory:"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(payload.query)
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
        return ToolResult(output={"columns": columns, "rows": rows})

    def _run_postgres(self, payload: SqlConnectorInput) -> ToolResult:
        try:
            import psycopg  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise ToolError(
                "Install agentforge[postgres] to enable Postgres connectivity."
            ) from exc
        if not payload.database:
            raise ToolError("Postgres requires a connection string in 'database'.")
        with psycopg.connect(payload.database) as conn:
            with conn.cursor() as cursor:
                cursor.execute(payload.query)
                rows = cursor.fetchall()
                columns = [col.name for col in cursor.description] if cursor.description else []
        return ToolResult(output={"columns": columns, "rows": rows})
