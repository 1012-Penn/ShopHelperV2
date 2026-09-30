"""Transactional MySQL repository for knowledge, staging, and extraction cursors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    FAQ,
    Conversation,
    KnowledgeChunk,
    KnowledgeExtractionCursor,
    KnowledgeQAStaging,
    Message,
)
from app.services.knowledge.content import (
    ChunkDraft,
    build_embedding_text,
    build_knowledge_fingerprint,
)


@dataclass(frozen=True)
class ConversationTurn:
    user_message_id: int
    assistant_message_id: int
    user_text: str
    assistant_text: str


class KnowledgeRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    @staticmethod
    def faq_drafts(faqs: list[FAQ]) -> list[ChunkDraft]:
        return [
            ChunkDraft(
                source_key=f"faq:{faq.id}",
                category=faq.category,
                questions=[faq.question],
                answer=faq.answer,
                chapter_path=[faq.category],
                content_type="product_faq",
                is_critical=False,
            )
            for faq in faqs
        ]

    def load_faq_drafts(self) -> list[ChunkDraft]:
        with self.session_factory() as session:
            faqs = list(session.scalars(select(FAQ).order_by(FAQ.id)))
            return self.faq_drafts(faqs)

    def upsert_drafts(self, drafts: list[ChunkDraft]) -> list[int]:
        if not drafts:
            return []
        with self.session_factory.begin() as session:
            ids = self._upsert_drafts(session, drafts)
            return ids

    @staticmethod
    def _upsert_drafts(session: Session, drafts: list[ChunkDraft]) -> list[int]:
        source_keys = {draft.source_key for draft in drafts}
        existing = {
            chunk.source_key: chunk
            for chunk in session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.source_key.in_(source_keys)))
        }
        rows: dict[str, KnowledgeChunk] = {}
        for draft in drafts:
            embedding_text = build_embedding_text(draft.category, draft.questions, draft.answer)
            content_hash = build_knowledge_fingerprint(draft.category, draft.questions, draft.answer)
            chunk = existing.get(draft.source_key)
            if chunk is None:
                chunk = KnowledgeChunk(
                    source_key=draft.source_key,
                    category=draft.category,
                    questions=draft.questions,
                    answer=draft.answer,
                    embedding_text=embedding_text,
                    chapter_path=draft.chapter_path,
                    content_type=draft.content_type,
                    is_critical=draft.is_critical,
                    vector_status="pending",
                    content_hash=content_hash,
                )
                session.add(chunk)
            else:
                if chunk.embedding_text != embedding_text:
                    chunk.embedding_text = embedding_text
                    chunk.vector_status = "pending"
                    chunk.vector_id = None
                chunk.category = draft.category
                chunk.questions = draft.questions
                chunk.answer = draft.answer
                chunk.chapter_path = draft.chapter_path
                chunk.content_type = draft.content_type
                chunk.is_critical = draft.is_critical
                chunk.content_hash = content_hash
            rows[draft.source_key] = chunk
        session.flush()

        neighbor_keys = {
            key
            for draft in drafts
            for key in (draft.previous_source_key, draft.next_source_key)
            if key is not None
        }
        neighbors = {
            chunk.source_key: chunk
            for chunk in session.scalars(
                select(KnowledgeChunk).where(KnowledgeChunk.source_key.in_(neighbor_keys))
            )
        } if neighbor_keys else {}
        neighbors.update(rows)
        for draft in drafts:
            chunk = rows[draft.source_key]
            previous = neighbors.get(draft.previous_source_key) if draft.previous_source_key else None
            following = neighbors.get(draft.next_source_key) if draft.next_source_key else None
            chunk.previous_chunk_id = previous.id if previous else None
            chunk.next_chunk_id = following.id if following else None
        session.flush()
        return [rows[draft.source_key].id for draft in drafts]

    def pending(self, limit: int) -> list[KnowledgeChunk]:
        if limit < 1:
            return []
        with self.session_factory() as session:
            return list(
                session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.vector_status == "pending")
                    .order_by(KnowledgeChunk.id)
                    .limit(limit)
                )
            )

    def mark_vectorized(self, chunk_id: int, vector_id: int) -> None:
        with self.session_factory.begin() as session:
            chunk = session.get(KnowledgeChunk, chunk_id)
            if chunk is None:
                raise LookupError(f"knowledge chunk {chunk_id} does not exist")
            chunk.vector_id = vector_id
            chunk.vector_status = "vectorized"

    def load_by_ids(self, ids: list[int]) -> list[KnowledgeChunk]:
        if not ids:
            return []
        with self.session_factory() as session:
            rows = list(session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(set(ids)))))
            return sorted(rows, key=lambda row: ids.index(row.id))

    def load_turns_after(self, last_message_id: int, limit: int) -> tuple[list[ConversationTurn], int]:
        if limit < 1:
            return [], last_message_id
        with self.session_factory() as session:
            messages = list(
                session.scalars(
                    select(Message)
                    .where(Message.id > last_message_id)
                    .order_by(Message.id)
                    .limit(limit)
                )
            )
            if not messages:
                return [], last_message_id
            next_cursor = messages[-1].id
            pending_user: dict[str, Message] = {}
            turns: list[ConversationTurn] = []
            for message in messages:
                if message.role == "user":
                    pending_user[message.conversation_id] = message
                elif message.role == "assistant":
                    user_message = pending_user.get(message.conversation_id)
                    if user_message is None:
                        continue
                    if message.tool_calls:
                        continue
                    pending_user.pop(message.conversation_id, None)
                    if message.content.strip():
                        turns.append(
                            ConversationTurn(
                                user_message_id=user_message.id,
                                assistant_message_id=message.id,
                                user_text=user_message.content,
                                assistant_text=message.content,
                            )
                        )
            if pending_user:
                # Keep the trailing unanswered user message in the next batch.
                next_cursor = min(next_cursor, min(message.id for message in pending_user.values()) - 1)
            return turns, next_cursor

    def stage_batch_and_advance(
        self,
        candidates: list[tuple[ConversationTurn, Any]],
        last_message_id: int,
        run_id: str,
    ) -> None:
        with self.session_factory.begin() as session:
            seen: set[tuple[str, int]] = set()
            for turn, candidate in candidates:
                questions = list(candidate.questions)
                fingerprint = build_knowledge_fingerprint(candidate.category, questions, candidate.answer)
                source_key = (fingerprint, turn.user_message_id)
                if source_key in seen:
                    continue
                seen.add(source_key)
                exists = session.scalar(
                    select(KnowledgeQAStaging.id).where(
                        KnowledgeQAStaging.fingerprint == fingerprint,
                        KnowledgeQAStaging.source_user_message_id == turn.user_message_id,
                    )
                )
                if exists is not None:
                    continue
                session.add(
                    KnowledgeQAStaging(
                        source_user_message_id=turn.user_message_id,
                        source_assistant_message_id=turn.assistant_message_id,
                        category=candidate.category,
                        questions=questions,
                        answer=candidate.answer,
                        fingerprint=fingerprint,
                        run_id=run_id,
                    )
                )
            cursor = session.get(KnowledgeExtractionCursor, "conversation_knowledge")
            if cursor is None:
                session.add(
                    KnowledgeExtractionCursor(cursor_name="conversation_knowledge", last_message_id=last_message_id)
                )
            elif last_message_id > cursor.last_message_id:
                cursor.last_message_id = last_message_id

    def promote_staged(self, run_id: str) -> tuple[int, int]:
        with self.session_factory.begin() as session:
            staged = list(
                session.scalars(
                    select(KnowledgeQAStaging)
                    .where(KnowledgeQAStaging.run_id == run_id)
                    .order_by(KnowledgeQAStaging.id)
                )
            )
            if not staged:
                return 0, 0
            fingerprints = {row.fingerprint for row in staged}
            existing = set(
                session.scalars(
                    select(KnowledgeChunk.content_hash).where(KnowledgeChunk.content_hash.in_(fingerprints))
                )
            )
            seen = set(existing)
            drafts: list[ChunkDraft] = []
            deduped = 0
            for row in staged:
                if row.fingerprint in seen:
                    deduped += 1
                    continue
                seen.add(row.fingerprint)
                drafts.append(
                    ChunkDraft(
                        source_key=f"conversation:{row.fingerprint}",
                        category=row.category,
                        questions=row.questions,
                        answer=row.answer,
                        chapter_path=[row.category],
                        content_type="conversation_qa",
                        is_critical=False,
                    )
                )
            self._upsert_drafts(session, drafts)
            return deduped, len(drafts)
