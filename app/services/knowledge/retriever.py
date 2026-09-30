"""Dense semantic FAQ retrieval backed by Milvus and authoritative MySQL rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.services.knowledge.vector_store import VectorHit


@dataclass(frozen=True)
class FAQHit:
    question: str
    answer: str
    category: str
    score: float


class DenseSearcher:
    def __init__(self, embeddings: Any, vector_store: Any):
        self.embeddings = embeddings
        self.vector_store = vector_store

    def search(self, query: str, limit: int) -> list[VectorHit]:
        vector = self.embeddings.embed_query(query)
        return self.vector_store.search(vector, limit=limit)


class KnowledgeRetriever:
    def __init__(self, searcher: Any, repository: Any, top_k: int, min_similarity: float):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
            raise ValueError("min_similarity must be a finite COSINE score")
        self.searcher = searcher
        self.repository = repository
        self.top_k = min(top_k, 5)
        self.min_similarity = min_similarity

    def search(self, query: str) -> list[FAQHit]:
        if not query.strip():
            return []
        vector_hits = self.searcher.search(query, limit=self.top_k)
        accepted = [hit for hit in vector_hits if hit.score >= self.min_similarity][: self.top_k]
        if not accepted:
            return []

        rows = self.repository.load_by_ids([hit.chunk_id for hit in accepted])
        rows_by_id = {row.id: row for row in rows}
        result: list[FAQHit] = []
        for hit in accepted:
            row = rows_by_id.get(hit.chunk_id)
            if row is None:
                continue
            questions = row.questions or []
            question = next((item.strip() for item in questions if item and item.strip()), "")
            if not question:
                chapter_path = row.chapter_path or []
                question = chapter_path[-1].strip() if chapter_path and chapter_path[-1].strip() else row.category
            if not question or not row.answer.strip():
                continue
            result.append(FAQHit(question, row.answer, row.category, hit.score))
        return result[:5]
