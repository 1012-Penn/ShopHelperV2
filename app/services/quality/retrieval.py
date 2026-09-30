"""Shared retrieval interface and authoritative evidence hydration."""
from dataclasses import dataclass, asdict, replace
from sqlalchemy import select
from app.db.models import KnowledgeChunk, HybridSync
from app.services.quality.query import QueryNormalizer


@dataclass(frozen=True)
class Evidence:
    n: int
    chunk_id: int
    section_path: list[str]
    question: str
    answer: str
    source_key: str
    score: float
    category: str = ''

    def snapshot(self):
        from app.services.quality.sources import source_url
        return {**asdict(self), "source_url": source_url(self.source_key, self.section_path)}


class QualityRetriever:
    def __init__(self, session_factory, embeddings, store, reranker, normalizer=None):
        self.session_factory=session_factory
        self.embeddings=embeddings
        self.store=store
        self.reranker=reranker
        self.normalizer=normalizer or QueryNormalizer()

    def hydrate(self, hits):
        with self.session_factory() as s:
            rows={r.id:r for r in s.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_([h.chunk_id for h in hits]),KnowledgeChunk.is_active.is_(True)))}
            synced={r.chunk_id:r.content_hash for r in s.scalars(select(HybridSync).where(HybridSync.collection==self.store.collection_name,HybridSync.chunk_id.in_(rows)))}
        accepted=[]
        for hit in hits:
            row=rows.get(hit.chunk_id)
            if row is None or row.content_hash!=hit.content_hash or synced.get(row.id)!=row.content_hash:
                continue
            accepted.append(Evidence(len(accepted)+1,row.id,row.chapter_path,(row.questions or [''])[0],row.answer,row.source_key,hit.score,row.category))
        return accepted

    def retrieve_with_candidates(self, query, strategy='hybrid_rerank', category=None):
        understanding=self.normalizer.normalize(query)
        vector=None if strategy=='bm25' else self.embeddings.embed_query(understanding.canonical)
        hits=self.store.search(vector,understanding.lexical,strategy,category=category,limit=50)
        candidates=self.hydrate(hits)
        if strategy=='hybrid_rerank' and candidates:
            ranks=self.reranker.rank(understanding.canonical,[e.question+'\n'+e.answer for e in candidates],10)
            evidence=[replace(candidates[i],n=n+1,score=score) for n,(i,score) in enumerate(ranks)]
        else:
            evidence=[replace(e,n=i+1) for i,e in enumerate(candidates[:10])]
        return evidence,candidates

    def retrieve(self, query, strategy='hybrid_rerank', category=None):
        return self.retrieve_with_candidates(query,strategy,category)[0]
