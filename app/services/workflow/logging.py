"""Append structured run traces without prompts, reasoning, or credentials."""
import json
from pathlib import Path
from threading import Lock
from datetime import datetime, timezone


class WorkflowLog:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = Lock()

    def write(self, state, status='done', error_type=None):
        record = {key: state.get(key) for key in (
            'run_id','conversation_id','path','intent','route','gate_passed','gate_score',
            'decisions','tool_calls','tool_trace','tokens','usage','stop_reason','message_id')}
        record.update(status=status, error_type=error_type,
                      occurred_at=datetime.now(timezone.utc).isoformat(),
                      evidence_ids=[e.get('chunk_id') for e in state.get('evidence', [])],
                      retrieval_trace=state.get('retrieval_trace', {}))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as file:
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
