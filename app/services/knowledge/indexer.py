"""Offline source ingestion and vector indexing coordination."""

from dataclasses import dataclass
from typing import Any

from app.services.knowledge.content import ChunkDraft
from app.services.knowledge.repository import KnowledgeRepository
from app.services.knowledge.vector_store import VectorRow


@dataclass(frozen=True)
class SyncSummary:
    pending_before: int
    vectorized: int
    failed: int
from app.services.knowledge.repository import KnowledgeRepository


class KnowledgeIndexer:
    def __init__(self, repository: KnowledgeRepository, embeddings: Any | None = None, vector_store: Any | None = None):
        self.repository = repository
        self.embeddings = embeddings
        self.vector_store = vector_store

    def import_drafts(self, drafts: list[ChunkDraft]) -> list[int]:
        """Persist complete source content before any vector service call."""
        return self.repository.upsert_drafts(drafts)

    def import_faqs(self) -> list[int]:
        """Copy the current authoritative FAQ rows into pending knowledge chunks."""
        return self.repository.upsert_drafts(self.repository.load_faq_drafts())

    def sync_pending(self, batch_size: int) -> SyncSummary:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.embeddings is None or self.vector_store is None:
            raise RuntimeError("knowledge embedding and vector services are not configured")
        pending = self.repository.pending(batch_size)
        summary = SyncSummary(pending_before=len(pending), vectorized=0, failed=0)
        if not pending:
            return summary
        try:
            self.vector_store.ensure_collection()
        except Exception:
            return SyncSummary(len(pending), 0, len(pending))

        try:
            vectors = self.embeddings.embed_documents([chunk.embedding_text for chunk in pending])
            if len(vectors) != len(pending) or any(len(vector) != 1024 for vector in vectors):
                raise ValueError("embedding response count or dimension mismatch")
        except Exception:
            successful = 0
            failed = 0
            for chunk in pending:
                try:
                    self._sync_one(chunk)
                    successful += 1
                except Exception:
                    failed += 1
            return SyncSummary(len(pending), successful, failed)

        rows = [VectorRow(chunk_id=chunk.id, vector=vector) for chunk, vector in zip(pending, vectors)]
        try:
            vector_ids = self.vector_store.upsert(rows)
            if vector_ids != [chunk.id for chunk in pending]:
                raise RuntimeError("vector store returned IDs in an unexpected order")
        except Exception:
            successful = 0
            failed = 0
            for chunk in pending:
                try:
                    self._sync_one(chunk)
                    successful += 1
                except Exception:
                    failed += 1
            return SyncSummary(len(pending), successful, failed)

        successful = 0
        failed = 0
        for chunk, vector_id in zip(pending, vector_ids):
            try:
                self.repository.mark_vectorized(chunk.id, vector_id)
                successful += 1
            except Exception:
                # MySQL remains pending; the next run upserts this same PK again.
                failed += 1
        return SyncSummary(len(pending), successful, failed)

    def _sync_one(self, chunk) -> None:
        vector = self.embeddings.embed_query(chunk.embedding_text)
        if len(vector) != 1024:
            raise ValueError("embedding response must have 1024 dimensions")
        vector_id = self.vector_store.upsert([VectorRow(chunk_id=chunk.id, vector=vector)])[0]
        self.repository.mark_vectorized(chunk.id, vector_id)

    def close(self) -> None:
        for service in (self.embeddings, self.vector_store):
            close = getattr(service, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        engine = self.repository.session_factory.kw.get("bind")
        if engine is not None:
            engine.dispose()
