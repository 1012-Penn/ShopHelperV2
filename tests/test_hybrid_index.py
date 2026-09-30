import pytest
from sqlalchemy import select
from app.db.models import KnowledgeChunk, HybridSync
from app.services.quality.index import HybridIndexer


def chunk():
    return KnowledgeChunk(source_key='doc:x:0',category='耳机',questions=['型号'],answer='原文',embedding_text='型号 原文',chapter_path=['耳机'],content_type='policy',content_hash='v1')


class Embeddings:
    def embed_documents(self,texts):
        return [[0.]*1024 for t in texts]


class Store:
    collection_name='test_hybrid'
    def __init__(self):
        self.rows={}
        self.failure=False
    def ensure_collection(self):
        pass
    def upsert(self,rows):
        self.rows.update({r.chunk_id:r for r in rows})
        if self.failure:
            raise RuntimeError('interrupted after Milvus')
        return [r.chunk_id for r in rows]
    def delete(self,ids):
        for id in ids:
            self.rows.pop(id,None)


def test_interrupted_sync_resumes_and_deletion_reconciles(db_session_factory):
    with db_session_factory.begin() as s:
        s.add(chunk())
    store=Store();store.failure=True
    indexer=HybridIndexer(db_session_factory,Embeddings(),store)
    with pytest.raises(RuntimeError):
        indexer.sync()
    with db_session_factory() as s:
        assert list(s.scalars(select(HybridSync)))==[]
    store.failure=False
    assert indexer.sync()['synchronized']==1
    assert indexer.sync()['synchronized']==0
    assert len(store.rows)==1
    with db_session_factory.begin() as s:
        s.scalar(select(KnowledgeChunk)).is_active=False
    assert indexer.sync()['deleted']==1
    assert store.rows=={}


def test_rebuild_cli_safe_failure(monkeypatch,capsys):
    from scripts import rebuild_hybrid_index
    def fail():
        raise RuntimeError('sk-secret')
    monkeypatch.setattr(rebuild_hybrid_index,'build_indexer',fail)
    assert rebuild_hybrid_index.main([])==1
    assert 'sk-secret' not in capsys.readouterr().out
