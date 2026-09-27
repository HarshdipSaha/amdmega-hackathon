"""Append-only JSONL record store. Content-addressed, so resume is free."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from silentpath.record import RunRecord


class RecordStore:
    def __init__(self, directory: Path | str, filename: str = "records.jsonl") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / filename
        self._ids: set[str] | None = None

    def _known_ids(self) -> set[str]:
        if self._ids is None:
            self._ids = {r.record_id for r in self.load()}
        return self._ids

    def has(self, record_id: str) -> bool:
        return record_id in self._known_ids()

    def append(self, record: RunRecord) -> bool:
        """Append unless already present. Returns True if written."""
        if self.has(record.record_id):
            return False
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        self._known_ids().add(record.record_id)
        return True

    def load(self) -> list[RunRecord]:
        if not self.path.exists():
            return []
        out: list[RunRecord] = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(RunRecord.model_validate_json(line))
                except (ValueError, json.JSONDecodeError):
                    continue   # an interrupted run leaves a truncated last line
        return out

    def by_workload(self) -> dict[str, list[RunRecord]]:
        groups: dict[str, list[RunRecord]] = defaultdict(list)
        for r in self.load():
            groups[r.workload_id].append(r)
        return dict(groups)
