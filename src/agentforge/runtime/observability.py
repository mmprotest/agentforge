"""Simple metrics collection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import time
from typing import Iterator


@dataclass
class MetricsCollector:
    workspace_dir: Path | None = None
    counters: dict[str, int] = field(default_factory=dict)
    timers: dict[str, list[float]] = field(default_factory=dict)

    def inc(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timers.setdefault(name, []).append(elapsed)

    def export_json(self) -> dict[str, object]:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counters": dict(self.counters),
            "timers": {key: list(values) for key, values in self.timers.items()},
        }
        if self.workspace_dir:
            metrics_dir = self.workspace_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            file_path = metrics_dir / f"{datetime.now(timezone.utc).date()}.jsonl"
            with file_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def write_run_summary(self, payload: dict[str, object]) -> None:
        if not self.workspace_dir:
            return
        metrics_dir = self.workspace_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        file_path = metrics_dir / f"{datetime.now(timezone.utc).date()}.jsonl"
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
