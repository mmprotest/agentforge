"""Audit logging with hash chaining."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from agentforge.runtime.storage import AuditStore


REDACT_KEYS = ("key", "token", "password", "secret")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def redact(payload: Any, rules: Iterable[str] = REDACT_KEYS) -> Any:
    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            if any(rule in key.lower() for rule in rules):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = redact(value, rules)
        return redacted
    if isinstance(payload, list):
        return [redact(item, rules) for item in payload]
    if isinstance(payload, str) and len(payload) > 2000:
        return payload[:2000] + "...[truncated]"
    return payload


@dataclass
class AuditEvent:
    timestamp: str
    run_id: str
    trace_id: str
    event_type: str
    payload: dict[str, Any]
    payload_hash: str
    prev_hash: str
    event_hash: str


class AuditLogger:
    def __init__(
        self,
        workspace_dir: Path,
        store: AuditStore | None = None,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.store = store
        self._prev_hashes: dict[str, str] = {}

    def emit(
        self,
        workspace_id: str,
        run_id: str,
        trace_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> AuditEvent:
        timestamp = datetime.now(timezone.utc).isoformat()
        safe_payload = redact(payload)
        payload_hash = sha256(canonical_json(safe_payload).encode("utf-8")).hexdigest()
        prev_hash = self._prev_hashes.get(run_id, "")
        event_hash = sha256(
            (prev_hash + payload_hash + event_type + timestamp).encode("utf-8")
        ).hexdigest()
        event = AuditEvent(
            timestamp=timestamp,
            run_id=run_id,
            trace_id=trace_id,
            event_type=event_type,
            payload=safe_payload,
            payload_hash=payload_hash,
            prev_hash=prev_hash,
            event_hash=event_hash,
        )
        self._prev_hashes[run_id] = event_hash
        self._write_jsonl(workspace_id, event)
        if self.store:
            self.store.append_event(
                {
                    "workspace_id": workspace_id,
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "event_type": event_type,
                    "timestamp": timestamp,
                    "payload": safe_payload,
                    "payload_hash": payload_hash,
                    "prev_hash": prev_hash,
                    "event_hash": event_hash,
                }
            )
        return event

    def _write_jsonl(self, workspace_id: str, event: AuditEvent) -> None:
        audit_dir = self.workspace_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        file_path = audit_dir / f"{event.run_id}.jsonl"
        record = {
            "workspace_id": workspace_id,
            "timestamp": event.timestamp,
            "run_id": event.run_id,
            "trace_id": event.trace_id,
            "event_type": event.event_type,
            "payload": event.payload,
            "payload_hash": event.payload_hash,
            "prev_hash": event.prev_hash,
            "event_hash": event.event_hash,
        }
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
