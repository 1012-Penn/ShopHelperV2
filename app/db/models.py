"""Database models for FAQ, conversations, messages, and support tickets."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class FAQ(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())


class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")
    tickets: Mapped[list["Ticket"]] = relationship(back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'tool')", name="ck_messages_role"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.conversation_id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tool_calls: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.conversation_id"), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    conversation: Mapped[Conversation] = relationship(back_populates="tickets")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        CheckConstraint(
            "vector_status IN ('pending', 'vectorized')",
            name="ck_knowledge_chunks_vector_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    questions: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    chapter_path: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False)
    is_critical: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    previous_chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True
    )
    vector_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    vector_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class KnowledgeQAStaging(Base):
    __tablename__ = "knowledge_qa_staging"
    __table_args__ = (
        UniqueConstraint(
            "fingerprint", "source_user_message_id", name="uq_knowledge_stage_fingerprint_source"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_user_message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    source_assistant_message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    category: Mapped[str] = mapped_column(Text, nullable=False)
    questions: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())


class KnowledgeExtractionCursor(Base):
    __tablename__ = "knowledge_extraction_cursor"

    cursor_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_message_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )
