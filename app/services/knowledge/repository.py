"""Transactional MySQL repository for knowledge, staging, and extraction cursors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    FAQ,
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
                if chunk.embedding_text != embedding_text or not chunk.is_active:
                    chunk.embedding_text = embedding_text
                    chunk.vector_status = "pending"
                    chunk.vector_id = None
                chunk.category = draft.category
                chunk.questions = draft.questions
                chunk.answer = draft.answer
                chunk.chapter_path = draft.chapter_path
                chunk.content_type = draft.content_type
                chunk.is_critical = draft.is_critical
                chunk.is_active = True
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
                    .where(KnowledgeChunk.vector_status == "pending", KnowledgeChunk.is_active.is_(True))
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
            rows = list(
                session.scalars(
                    select(KnowledgeChunk).where(
                        KnowledgeChunk.id.in_(set(ids)), KnowledgeChunk.is_active.is_(True)
                    )
                )
            )
            return sorted(rows, key=lambda row: ids.index(row.id))

    def deactivate_missing_documents(
        self, active_sources: list[str], retained_source_keys: list[str]
    ) -> list[tuple[int, int]]:
        active = set(active_sources)
        retained = set(retained_source_keys)
        with self.session_factory.begin() as session:
            document_chunks = list(
                session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.source_key.startswith("doc:")))
            )
            for chunk in document_chunks:
                source = chunk.source_key[4:].rsplit(":", 1)[0]
                if source not in active or chunk.source_key not in retained:
                    chunk.is_active = False
            return [
                (chunk.id, chunk.vector_id)
                for chunk in document_chunks
                if not chunk.is_active and chunk.vector_id is not None
            ]

    def clear_deleted_vector_ids(self, chunk_ids: list[int]) -> None:
        if not chunk_ids:
            return
        with self.session_factory.begin() as session:
            rows = list(session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(set(chunk_ids)))))
            for row in rows:
                if not row.is_active:
                    row.vector_id = None

    def load_turns_after(
        self, last_message_id: int, limit: int, conversation_id: str
    ) -> tuple[list[ConversationTurn], int]:
        if limit < 1:
            return [], last_message_id
        with self.session_factory() as session:
            user_query = select(Message).where(
                Message.id > last_message_id,
                Message.role == "user",
                Message.conversation_id == conversation_id,
            )
            users = list(session.scalars(user_query.order_by(Message.id).limit(limit)))
            if not users:
                latest_query = select(func.max(Message.id)).where(
                    Message.id > last_message_id,
                    Message.conversation_id == conversation_id,
                )
                latest_id = session.scalar(latest_query)
                return [], latest_id or last_message_id
            next_cursor = last_message_id
            turns: list[ConversationTurn] = []
            for user_message in users:
                next_user_id = session.scalar(
                    select(Message.id)
                    .where(
                        Message.conversation_id == user_message.conversation_id,
                        Message.role == "user",
                        Message.id > user_message.id,
                    )
                    .order_by(Message.id)
                    .limit(1)
                )
                assistant_query = select(Message).where(
                    Message.conversation_id == user_message.conversation_id,
                    Message.role == "assistant",
                    Message.id > user_message.id,
                )
                if next_user_id is not None:
                    assistant_query = assistant_query.where(Message.id < next_user_id)
                assistants = list(session.scalars(assistant_query.order_by(Message.id)))
                final_answer = next(
                    (message for message in assistants if not message.tool_calls and message.content.strip()),
                    None,
                )
                if final_answer is not None:
                    turns.append(
                        ConversationTurn(
                            user_message_id=user_message.id,
                            assistant_message_id=final_answer.id,
                            user_text=user_message.content,
                            assistant_text=final_answer.content,
                        )
                    )
                    next_cursor = max(next_cursor, final_answer.id)
                elif next_user_id is not None:
                    next_cursor = max(next_cursor, next_user_id - 1)
                else:
                    # Keep the trailing unanswered turn for a later scheduled run.
                    break
            if next_cursor == last_message_id:
                return [], last_message_id
            return turns, next_cursor

    def extraction_cursor(self, conversation_id: str | None = None) -> int:
        cursor_name = (
            "conversation_knowledge"
            if conversation_id is None
            else self._conversation_cursor_name(conversation_id)
        )
        with self.session_factory() as session:
            cursor = session.get(KnowledgeExtractionCursor, cursor_name)
            return cursor.last_message_id if cursor else 0

    @staticmethod
    def _conversation_cursor_name(conversation_id: str) -> str:
        return hashlib.sha256(f"conversation:{conversation_id}".encode()).hexdigest()

    def conversations_with_pending_turns(self) -> list[tuple[str, int]]:
        with self.session_factory() as session:
            latest_user_ids = list(
                session.execute(
                    select(Message.conversation_id, func.max(Message.id))
                    .where(Message.role == "user")
                    .group_by(Message.conversation_id)
                    .order_by(func.max(Message.id))
                )
            )
            cursor_names = {
                conversation_id: self._conversation_cursor_name(conversation_id)
                for conversation_id, _ in latest_user_ids
            }
            stored_cursors = {
                cursor_name: last_message_id
                for cursor_name, last_message_id in session.execute(
                    select(KnowledgeExtractionCursor.cursor_name, KnowledgeExtractionCursor.last_message_id)
                )
            }
            return [
                (conversation_id, stored_cursors.get(cursor_names[conversation_id], 0))
                for conversation_id, latest_user_id in latest_user_ids
                if latest_user_id > stored_cursors.get(cursor_names[conversation_id], 0)
            ]

    def count_messages_between(self, after_id: int, through_id: int, conversation_id: str) -> int:
        if through_id <= after_id:
            return 0
        with self.session_factory() as session:
            query = select(func.count()).select_from(Message).where(
                Message.id > after_id,
                Message.id <= through_id,
            )
            query = query.where(Message.conversation_id == conversation_id)
            return session.scalar(query) or 0

    def count_skipped_tool_messages(
        self, after_id: int, through_id: int, conversation_id: str
    ) -> int:
        if through_id <= after_id:
            return 0
        with self.session_factory() as session:
            query = select(Message).where(Message.id > after_id, Message.id <= through_id)
            query = query.where(Message.conversation_id == conversation_id)
            messages = list(session.scalars(query))
            return sum(message.role == "tool" or bool(message.tool_calls) for message in messages)

    def stage_batch_and_advance(
        self,
        candidates: list[tuple[ConversationTurn, Any]],
        last_message_id: int,
        run_id: str,
        conversation_id: str,
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
            cursor_name = self._conversation_cursor_name(conversation_id)
            cursor = session.get(KnowledgeExtractionCursor, cursor_name)
            if cursor is None:
                session.add(
                    KnowledgeExtractionCursor(cursor_name=cursor_name, last_message_id=last_message_id)
                )
            elif last_message_id > cursor.last_message_id:
                cursor.last_message_id = last_message_id

    def promote_staged(self) -> tuple[int, int]:
        with self.session_factory.begin() as session:
            staged = list(
                session.scalars(
                    select(KnowledgeQAStaging)
                    .where(KnowledgeQAStaging.promoted_at.is_(None))
                    .order_by(KnowledgeQAStaging.id)
                )
            )
            if not staged:
                return 0, 0
            fingerprints = {row.fingerprint for row in staged}
            existing = set(
                session.scalars(
                    select(KnowledgeChunk.content_hash)
                    .where(KnowledgeChunk.content_hash.in_(fingerprints))
                    .where(KnowledgeChunk.is_active.is_(True))
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
            for row in staged:
                row.promoted_at = func.now()
            return deduped, len(drafts)
