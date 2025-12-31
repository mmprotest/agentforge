from pathlib import Path
import sqlite3

from agentforge.tools.connectors.filesystem import FilesystemConnectorTool
from agentforge.tools.connectors.sql import SqlConnectorTool


def test_filesystem_connector_reads(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    data_dir = workspace / "data"
    data_dir.mkdir()
    target = data_dir / "note.txt"
    target.write_text("hello", encoding="utf-8")
    tool = FilesystemConnectorTool(str(workspace))
    result = tool.run({"action": "read", "path": "note.txt"})
    assert result.output["content"] == "hello"


def test_sql_connector_select(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO items VALUES (1, 'alpha')")
        conn.commit()
    tool = SqlConnectorTool(str(tmp_path))
    result = tool.run({"engine": "sqlite", "database": str(db_path), "query": "SELECT name FROM items"})
    assert result.output["rows"] == [("alpha",)]
