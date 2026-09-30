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


def test_overlap_cannot_overwrite_newer_index(tmp_path):
    from threading import Thread,Event
    from app.db.session import make_engine,create_tables,make_session_factory
    engine=make_engine('sqlite:///'+str(tmp_path/'race.db'));create_tables(engine)
    sessions=make_session_factory(engine)
    with sessions.begin() as s:s.add(chunk())
    started=Event();release=Event()
    class Blocking(Store):
        def upsert(self,rows):
            if rows[0].content_hash=='v1':
                started.set();release.wait(5)
            return super().upsert(rows)
    store=Blocking();one=HybridIndexer(sessions,Embeddings(),store);two=HybridIndexer(sessions,Embeddings(),store)
    errors=[]
    def run(index):
        try:index.sync()
        except Exception as e:errors.append(e)
    t=Thread(target=run,args=(one,));t.start();assert started.wait(2)
    with sessions.begin() as s:
        row=s.scalar(select(KnowledgeChunk));row.content_hash='v2';row.answer='v2'
    other=Thread(target=run,args=(two,));other.start()
    import time
    time.sleep(.15);release.set();t.join(5);other.join(5)
    assert not errors
    assert store.rows[1].content_hash=='v2'
    assert one.sync()['synchronized']==0


def test_legacy_sync_updates_enabled_hybrid(db_session_factory):
    from app.services.knowledge.indexer import KnowledgeIndexer
    from app.services.knowledge.repository import KnowledgeRepository
    from types import SimpleNamespace
    class Legacy:
        def upsert(self,rows):return [r.chunk_id for r in rows]
    class Embedding:
        def embed_documents(self,texts):return [[0.]*1024 for x in texts]
    calls=[]
    class Hybrid:
        def sync(self,batch_size=32):calls.append(batch_size)
    indexer=KnowledgeIndexer(KnowledgeRepository(db_session_factory),Embedding(),Legacy())
    indexer.hybrid_indexer=Hybrid()
    indexer.sync_pending(8)
    assert calls==[8]
