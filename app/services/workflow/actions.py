"""Only a confirmed button request may invoke the existing ticket write tool."""

import json
import logging

from sqlalchemy import select

from app.db.models import Conversation, Message
from app.tools.business import build_tools


class TicketActions:
    def __init__(self, session_factory, runner_factory, locks):
        self.session_factory, self.runner_factory, self.locks = (
            session_factory,
            runner_factory,
            locks,
        )

    def submit(self, conversation_id, user_id, message_id):
        with self.locks.hold(conversation_id):
            with self.session_factory.begin() as s:
                conv = s.get(Conversation, conversation_id)
                row = s.scalar(
                    select(Message).where(Message.id == message_id).with_for_update()
                )
                if (
                    conv is None
                    or conv.user_id != user_id
                    or row is None
                    or row.conversation_id != conversation_id
                ):
                    raise ValueError("invalid conversation or suggestion")
                metadata = row.actions or {}
                if row.role != "assistant" or "create_ticket" not in metadata.get(
                    "items", []
                ):
                    raise ValueError("ticket was not suggested")
                ticket = metadata.get("ticket") or {"status": "offered"}
                if ticket["status"] != "offered":
                    return dict(ticket)
                description = metadata["question"]
                row.actions = {**metadata, "ticket": {"status": "submitting"}}
            # The reservation above commits before invoking any write tool.
            runner = self.runner_factory(
                [
                    t
                    for t in build_tools(self.session_factory, conversation_id)
                    if t.name == "create_ticket"
                ]
            )
            outcome = {
                "status": "unknown",
                "message": "工单提交结果暂未确认，请勿重复提交；请联系人工客服核查。",
            }
            try:
                result = runner.run(
                    "create_ticket",
                    {"description": description, "ticket_type": "客服跟进"},
                    f"ticket-{message_id}",
                )
                if not result.is_error:
                    parsed = json.loads(result.content)
                    if isinstance(parsed.get("ticket_no"), str) and parsed["ticket_no"]:
                        outcome = {
                            "status": "created",
                            "ticket_no": parsed["ticket_no"],
                        }
            except Exception as error:  # noqa: BLE001 - isolate external provider/tool failures
                logging.getLogger(__name__).warning(
                    "Ticket outcome unknown (%s)", type(error).__name__
                )
            finally:
                runner.close()
            with self.session_factory.begin() as s:
                row = s.get(Message, message_id)
                row.actions = {**row.actions, "ticket": outcome}
            return outcome
