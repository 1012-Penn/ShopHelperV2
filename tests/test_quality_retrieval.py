import httpx
import pytest
from app.services.quality.query import QueryNormalizer
from app.services.quality.rerank import Reranker
from app.services.quality.retrieval import QualityRetriever
from app.services.knowledge.hybrid_store import HybridHit
from app.db.models import KnowledgeChunk, HybridSync


def test_normalizer_preserves_model_negation_and_no_duplicate_documents():
    normalize=QueryNormalizer(lambda raw: '耳机支持快充吗')
    q=normalize.normalize('XH-300 不是快充吧？')
    assert 'XH-300' in q.canonical and '不' in q.canonical
    assert '邮费' in QueryNormalizer().normalize('邮费多少').lexical
    assert '运费' in QueryNormalizer().normalize('邮费多少').lexical


def test_reranker_rejects_invalid_index_and_nonfinite():
    def transport(request):
        return httpx.Response(200,json={'results':[{'index':9,'relevance_score':.9}]})
    ranker=Reranker('key',client=httpx.Client(transport=httpx.MockTransport(transport)))
    with pytest.raises(ValueError):
        ranker.rank('q',['a'],10)


def test_reranker_maps_index_order():
    def transport(request):
        return httpx.Response(200,json={'results':[{'index':1,'relevance_score':.9},{'index':0,'relevance_score':.3}]})
    ranker=Reranker('key',client=httpx.Client(transport=httpx.MockTransport(transport)))
    assert ranker.rank('q',['a','b'],10)==[(1,.9),(0,.3)]


def test_stale_hash_inactive_and_missing_hits_are_ignored(db_session_factory):
    with db_session_factory.begin() as s:
        s.add(KnowledgeChunk(id=7,source_key='doc:x:0',category='耳机',questions=['支持什么'],answer='USB-C',embedding_text='USB-C',chapter_path=['耳机','接口'],content_type='policy',content_hash='new'))
        s.flush()
        s.add(HybridSync(collection='hybrid',chunk_id=7,content_hash='new'))
    class Store:
        collection_name='hybrid'
    retriever=QualityRetriever(db_session_factory,None,Store(),None,QueryNormalizer())
    assert retriever.hydrate([HybridHit(7,.8,'old'),HybridHit(999,.9,'new')])==[]
    assert retriever.hydrate([HybridHit(7,.8,'new')])[0].section_path==['耳机','接口']


def test_normalization_reused_across_strategies():
    calls=[]
    def rewrite(raw):calls.append(raw);return raw
    normalizer=QueryNormalizer(rewrite)
    normalizer.normalize('邮费多少')
    normalizer.normalize('邮费多少')
    assert len(calls)==1
