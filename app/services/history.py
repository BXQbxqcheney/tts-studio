"""生成历史：JSONL 追加与最近记录读取。"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app import config
from app.schemas import HistoryItem

logger = logging.getLogger("tts_studio.history")


class HistoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.HISTORY_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, item: HistoryItem) -> None:
        line = item.model_dump_json()
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def list(self, limit: int = 5) -> list[HistoryItem]:
        limit = max(1, min(int(limit), 20))
        if not self.path.is_file():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        items: list[HistoryItem] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                items.append(HistoryItem.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                logger.warning("skip corrupt history line: %s", exc)
            if len(items) >= limit:
                break
        return items
