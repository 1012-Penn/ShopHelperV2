"""Offline source ingestion and vector indexing coordination."""

from app.services.knowledge.content import ChunkDraft
from app.services.knowledge.repository import KnowledgeRepository


class KnowledgeIndexer:
    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository

    def import_drafts(self, drafts: list[ChunkDraft]) -> list[int]:
        """Persist complete source content before any vector service call."""
        return self.repository.upsert_drafts(drafts)

    def import_faqs(self) -> list[int]:
        """Copy the current authoritative FAQ rows into pending knowledge chunks."""
        return self.repository.upsert_drafts(self.repository.load_faq_drafts())
