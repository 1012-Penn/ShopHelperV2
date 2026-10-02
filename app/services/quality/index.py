"""Recoverable collection synchronization keyed by current content hashes."""
from sqlalchemy import select, text
from contextlib import contextmanager
import hashlib
import tempfile
from pathlib import Path
from app.db.models import KnowledgeChunk, HybridSync
from app.services.knowledge.hybrid_store import HybridRow


class HybridIndexer:
    def __init__(self, session_factory, embeddings, store):
        self.session_factory, self.embeddings, self.store = session_factory, embeddings, store

    @contextmanager
    def _collection_lock(self):
        name='mewhelp-hybrid-'+hashlib.sha256(self.store.collection_name.encode()).hexdigest()[:32]
        engine=self.session_factory.kw['bind']
        if engine.dialect.name=='mysql':
            with engine.connect() as c:
                if c.execute(text('SELECT GET_LOCK(:name, 60)'),{'name':name}).scalar()!=1:
                    raise RuntimeError('hybrid collection sync lock unavailable')
                try:
                    yield
                finally:
                    c.execute(text('SELECT RELEASE_LOCK(:name)'),{'name':name})
        else:
            import fcntl
            with (Path(tempfile.gettempdir())/(name+'.lock')).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock,fcntl.LOCK_UN)

    def sync(self, batch_size=32):
        with self._collection_lock():
            return self._sync_locked(batch_size)

    def _sync_locked(self, batch_size):
        if batch_size < 1:
            raise ValueError('positive batch size required')
        self.store.ensure_collection()
        with self.session_factory() as s:
            rows=list(s.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)))
            done={r.chunk_id:r.content_hash for r in s.scalars(select(HybridSync).where(HybridSync.collection==self.store.collection_name))}
        active={r.id:r for r in rows if r.is_active}
        deleted=[id for id in done if id not in active]
        self.store.delete(deleted)
        with self.session_factory.begin() as s:
            for id in deleted:
                s.delete(s.get(HybridSync,(self.store.collection_name,id)))
        pending=[r for r in active.values() if done.get(r.id)!=r.content_hash]
        for start in range(0,len(pending),batch_size):
            batch=pending[start:start+batch_size]
            vectors=self.embeddings.embed_documents([r.embedding_text for r in batch])
            if len(vectors)!=len(batch):
                raise ValueError('embedding batch size mismatch')
            self.store.upsert([HybridRow(r.id,v,r.embedding_text,r.category,r.content_hash) for r,v in zip(batch,vectors)])
            with self.session_factory.begin() as s:
                for row in batch:
                    current=s.get(KnowledgeChunk,row.id)
                    if current and current.is_active and current.content_hash==row.content_hash:
                        s.merge(HybridSync(collection=self.store.collection_name,chunk_id=row.id,content_hash=row.content_hash))
        return {'active':len(active),'synchronized':len(pending),'deleted':len(deleted)}
