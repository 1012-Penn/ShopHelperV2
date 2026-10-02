"""Append-only snapshots for generation protocol failures."""
import json
from pathlib import Path
from threading import Lock


class JsonlGenerationFailureSink:
    """Write one JSON event per line, serializing concurrent in-process appends."""

    def __init__(self, path):
        self.path=Path(path)
        self._lock=Lock()

    def write(self, event):
        line=json.dumps(event,ensure_ascii=False,separators=(',',':'))
        with self._lock:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a',encoding='utf-8') as stream:
                stream.write(line+'\n')
