"""Server-owned ticket intent and pure checkpointed confirmation nodes."""

import re
from uuid import uuid4

from langgraph.types import interrupt

from app.db.models import TicketIntent


def ticket_request_cancelled(text):
    """Recognize a direct cancellation without treating quoted questions as one."""
    if not text or re.search(r'[“”"「」『』]|(?:他说|她说|客户说|客服说|用户说)', text):
        return False
    if re.search(
        r"(?:不要|不需要|不用|不想|别).{0,6}(?:取消|撤销).{0,8}(?:工单|建单)", text
    ):
        return False
    if re.search(r"(?:能否|能不能|是否|吗[？?]?$)", text):
        return False
    return bool(
        re.search(
            r"(?:不需要|不再需要|不用|不想|不要|别|不(?:再)?(?:建|创建|提交|开)|取消|撤销|停止).{0,12}(?:工单|建单)"
            r"|(?:工单|建单).{0,6}(?:取消|撤销|停止)",
            text,
        )
    )


def explicit_ticket_request(text):
    # Deliberately conservative: questions, quotations and negations never grant intent.
    if ticket_request_cancelled(text):
        return False
    if re.search(
        r'[“”"「」『』]|(不要|不想|别|不需要|不用|不建|取消|能否|能不能|可以吗|吗[？?]?$|怎么|如何|是否|教程|例如|比如|他说|她说|客户说|客服说|用户说)',
        text,
    ):
        return False
    return bool(
        re.search(
            r"(帮我|给我|请|我要|我想|麻烦|需要).{0,12}(建|创建|提交|开).{0,4}工单",
            text,
        )
    )


class TicketConfirmation:
    def __init__(self, service):
        self.service = service

    def intent(self, conversation_id, user_id, text=None):
        with self.service.session_factory.begin() as session:
            row = session.get(TicketIntent, conversation_id)
            if row is not None and row.user_id != user_id:
                raise ValueError("foreign ticket intent")
            if text and explicit_ticket_request(text):
                if row is None:
                    row = TicketIntent(conversation_id=conversation_id, user_id=user_id)
                    session.add(row)
                row.operation_id = str(uuid4())
                row.original_request = text
                row.status = "requested"
            elif text and ticket_request_cancelled(text) and row:
                row.status = "cancelled"
            if row and row.status == "requested":
                return {
                    "operation_id": row.operation_id,
                    "original_request": row.original_request,
                }
        return None

    def close_intent(self, conversation_id, status):
        with self.service.session_factory.begin() as session:
            row = session.get(TicketIntent, conversation_id)
            if row:
                row.status = status

    def offer(self, state):
        call = state["pending"]["calls"][state["tool_index"]]
        data = {
            "conversation_id": state["conversation_id"],
            "request_id": str(uuid4()),
            "tool_call_id": call["id"],
            "ticket_type": call["args"]["ticket_type"],
            "description": call["args"]["description"],
        }
        mid = self.service.store.save_answer(
            state["conversation_id"],
            "请确认工单信息。",
            [],
            [],
            state["question"],
            offer={"ticket_preview": data},
        )
        data["message_id"] = mid
        return {"ticket_preview": data, "ticket_decision": None}

    def await_confirmation(self, state):
        decision = interrupt(state["ticket_preview"])
        if not isinstance(decision, dict) or type(decision.get("approve")) is not bool:
            raise ValueError("Invalid confirmation")
        return {"ticket_decision": decision["approve"], "agent_next": "execute"}
