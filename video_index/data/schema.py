"""JSON Lines records. Result files are append-only, so a rerun continues after the last finished row; a key names
the fields that identify a row, and a later row with the same key replaces an earlier one for readers."""
from __future__ import annotations

import json
import os
import threading
from typing import Iterable


def read_jsonl(path: str) -> list[dict]:
    """Rows of a JSON Lines file; [] when the file does not exist. Blank lines and a line cut off by an interrupted
    write are skipped."""
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    """Write rows to path, replacing the file in one step."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class JsonlWriter:
    """Append rows to a JSON Lines file from several threads.

        with JsonlWriter(path, key=("qid", "role")) as w:
            if not w.has(row):
                w.write(row)

    ``has`` is true for rows whose key is in the file already or was written through this writer."""

    def __init__(self, path: str, key: str | tuple | list = "item_id"):
        self.path = path
        self.key = tuple(key) if isinstance(key, (tuple, list)) else (key,)
        self._lock = threading.Lock()
        self._seen = {self._key(r) for r in read_jsonl(path)}
        self._f = None

    def _key(self, row: dict) -> tuple:
        return tuple(row.get(k) for k in self.key)

    def __enter__(self) -> "JsonlWriter":
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def has(self, row: dict) -> bool:
        return self._key(row) in self._seen

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            self._f.write(line)
            self._f.flush()
            self._seen.add(self._key(row))

    def close(self) -> None:
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None
