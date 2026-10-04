"""Atomic summary append and monotonic anchors; never edits original messages."""

from sqlalchemy import func, select, update

from app.db.models import Conversation, ConversationSummary, Message


class ContextRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def read(self, cid):
        with self.session_factory() as s:
            meta = s.get(Conversation, cid)
            parts = list(
                s.scalars(
                    select(ConversationSummary)
                    .where(ConversationSummary.conversation_id == cid)
                    .order_by(ConversationSummary.segment)
                )
            )
            return meta, parts

    def original_users(self, cid):
        with self.session_factory() as s:
            return list(
                s.scalars(
                    select(Message)
                    .where(Message.conversation_id == cid, Message.role == "user")
                    .order_by(Message.id)
                )
            )

    def advance_layer1(self, cid, mid):
        with self.session_factory.begin() as s:
            s.execute(
                update(Conversation)
                .where(
                    Conversation.conversation_id == cid,
                    (Conversation.layer1_from_msg_id.is_(None))
                    | (Conversation.layer1_from_msg_id < mid),
                )
                .values(layer1_from_msg_id=mid)
            )

    def append(self, batch, content):
        with self.session_factory.begin() as s:
            result = s.execute(
                update(Conversation)
                .where(
                    Conversation.conversation_id == batch.conversation_id,
                    Conversation.summary_upto_msg_id == batch.previous_upto,
                )
                .values(summary_upto_msg_id=batch.upto_msg_id)
            )
            if result.rowcount != 1:
                return False
            segment = (
                s.scalar(
                    select(func.max(ConversationSummary.segment)).where(
                        ConversationSummary.conversation_id == batch.conversation_id
                    )
                )
                or 0
            ) + 1
            s.add(
                ConversationSummary(
                    conversation_id=batch.conversation_id,
                    segment=segment,
                    from_msg_id=batch.from_msg_id,
                    upto_msg_id=batch.upto_msg_id,
                    content=content,
                )
            )
            return segment
