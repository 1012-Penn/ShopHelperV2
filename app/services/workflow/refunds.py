"""Persistence and offer validation for local demo refund applications."""

from sqlalchemy.exc import IntegrityError

from app.db.models import Conversation, Message, RefundApplication
from app.services.workflow.demo_orders import get_demo_order


class RefundOfferError(ValueError):
    pass


class RefundConflict(ValueError):
    pass


class DemoOrderError(ValueError):
    pass


class DemoRefundApplications:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    @staticmethod
    def _same_request(saved, request):
        return all(
            getattr(saved, name) == getattr(request, name)
            for name in (
                "conversation_id", "message_id", "user_id", "order_id",
                "request_type", "reason",
            )
        )

    @staticmethod
    def _response(row):
        return {
            "status": "recorded",
            "request_id": row.request_id,
            "application_id": row.request_id,
            "message": "演示申请已记录；未执行真实退款或资金退回。",
        }

    def submit(self, request):
        order = get_demo_order(request.order_id, request.user_id)
        if order is None:
            raise DemoOrderError("unknown demo order")
        try:
            with self.session_factory.begin() as session:
                conversation = session.get(Conversation, request.conversation_id)
                if conversation is None or conversation.user_id != request.user_id:
                    raise RefundOfferError("conversation owner mismatch")
                message = session.get(Message, request.message_id)
                metadata = (message.actions or {}) if message else {}
                form = metadata.get("refund_form") or {}
                if (
                    message is None
                    or message.conversation_id != request.conversation_id
                    or message.role != "assistant"
                    or "refund_form" not in (metadata.get("items") or [])
                    or form.get("order_id") != request.order_id
                    or form.get("request_id") != request.request_id
                    or form.get("request_type") != request.request_type
                ):
                    raise RefundOfferError("refund form was not offered")

                existing = session.get(RefundApplication, request.request_id)
                if existing is not None:
                    if not self._same_request(existing, request):
                        raise RefundConflict("request id reused with different content")
                    return self._response(existing)
                row = RefundApplication(
                    request_id=request.request_id,
                    conversation_id=request.conversation_id,
                    message_id=request.message_id,
                    user_id=request.user_id,
                    order_id=request.order_id,
                    request_type=request.request_type,
                    reason=request.reason,
                    status="recorded",
                )
                session.add(row)
                session.flush()
                return self._response(row)
        except IntegrityError:
            # A concurrent retry can race the read above; the primary key is the arbiter.
            with self.session_factory() as session:
                saved = session.get(RefundApplication, request.request_id)
                if saved is None:
                    raise
                if not self._same_request(saved, request):
                    raise RefundConflict("request id reused with different content") from None
                return self._response(saved)
