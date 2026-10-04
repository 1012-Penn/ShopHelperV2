"""Local exact permissions; remote descriptions never authorize execution."""

import json
from pathlib import Path


class ToolPolicy:
    def __init__(self, path=None):
        self.path = Path(path) if path else None

    def permission(self, source, name):
        if source == "builtin":
            return "write" if name == "create_ticket" else "read"
        if not source.startswith("mcp:") or not self.path:
            return "deny"
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            permission = data["mcp"][source[4:]].get(name)
            return "read" if permission == "read" else "deny"
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return "deny"
