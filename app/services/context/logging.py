"""Append-only JSON lines with complete outbound context, readable with grep."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import convert_to_openai_messages


class ContextLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, event, **fields):
        row = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def context(self, event, messages, budget, **fields):
        self.write(
            event,
            messages=convert_to_openai_messages(messages),
            tokens=budget.count(messages) + budget.count(fields.get("tools", [])),
            window_count=len(messages),
            **fields,
        )
