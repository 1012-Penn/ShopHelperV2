"""Independent best-effort audit writes must not influence tool outcomes."""

import logging

from app.db.models import ToolAuditLog


class AuditWriter:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def record(self, result, args, context):
        try:
            with self.session_factory.begin() as session:
                session.add(
                    ToolAuditLog(
                        conversation_id=context.conversation_id,
                        tool_call_id=result.tool_call_id,
                        tool_name=result.tool_name,
                        source=result.source,
                        arguments=args,
                        result_summary=result.content[:4000],
                        status=result.status,
                        error=(result.error_detail or result.content)
                        if result.is_error
                        else None,
                        retry_count=result.retry_count,
                        duration_ms=result.duration_ms,
                    )
                )
        except Exception:
            logging.getLogger(__name__).exception("Tool audit write failed")
