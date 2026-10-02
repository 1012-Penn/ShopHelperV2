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


@pytest.mark.parametrize('raw,changed',[
 ('XH-300 支持七天退货吗','XH-300 支持三十天退货吗'),
 ('XH-300 不支持游泳但支持淋雨吗','XH-300 支持游泳但不支持淋雨吗')])
def test_normalizer_does_not_change_chinese_number_or_negative_scope(raw,changed):
    q=QueryNormalizer(lambda value:changed).normalize(raw)
    assert q.canonical==raw
    assert q.downgrade_reason


@pytest.mark.parametrize(('raw','candidate'),[
    ('MX-474分别有哪些接口？','MX-474有哪些接口？'),
    ('MX-474是否支持快充？','MX-474支持快充吗？'),
    ('MX-474这个型号能快一点吗？','MX-474这个型号速度能快些吗？'),
])
def test_normalizer_does_not_mistake_chinese_function_words_for_negation_or_numbers(raw,candidate):
    q=QueryNormalizer(lambda value:candidate).normalize(raw)
    assert q.canonical==candidate
    assert not q.downgrade_reason


def test_normalizer_allows_removing_a_repeated_occurrence_of_the_same_model():
    raw='MX-474Pro和MX-474Pro怎么选？'
    candidate='MX-474Pro怎么选？'
    q=QueryNormalizer(lambda value:candidate).normalize(raw)
    assert q.canonical==candidate
    assert not q.downgrade_reason


def test_normalizer_rejects_dropping_one_model_from_a_comparison():
    raw='MX-474Pro和MX-405-SE哪个更适合？'
    candidate='MX-474Pro哪个更适合？'
    q=QueryNormalizer(lambda value:candidate).normalize(raw)
    assert q.canonical==raw
    assert q.downgrade_reason


def test_normalizer_rejects_changing_an_arabic_amount():
    raw='MX-474差一点到220元'
    candidate='MX-474差一点到200元'
    q=QueryNormalizer(lambda value:candidate).normalize(raw)
    assert q.canonical==raw
    assert q.downgrade_reason


@pytest.mark.parametrize(('raw','candidate'),[
    ('MX-474售价一点五元','MX-474售价二点五元'),
    ('MX-474支持7天退货','MX-474支持7个月退货'),
])
def test_normalizer_preserves_chinese_and_arabic_quantities_with_units(raw,candidate):
    q=QueryNormalizer(lambda value:candidate).normalize(raw)
    assert q.canonical==raw
    assert q.downgrade_reason


def test_normalizer_allows_rewriting_a_positive_clause_when_negative_clause_is_unchanged():
    raw='MX-474不能游泳，能淋雨吗？'
    candidate='MX-474不能游泳，是否支持淋雨？'
    q=QueryNormalizer(lambda value:candidate).normalize(raw)
    assert q.canonical==candidate
    assert not q.downgrade_reason


def test_normalizer_rejects_a_candidate_that_adds_a_negative_scope():
    raw='MX-474支持快充吗？'
    candidate='MX-474不支持快充吗？'
    q=QueryNormalizer(lambda value:candidate).normalize(raw)
    assert q.canonical==raw
    assert q.downgrade_reason


@pytest.mark.parametrize(('raw','candidate'),[
    ('MX-474不同意退款','MX-474同意退款'),
    ('MX-474金额不过220元','MX-474金额超过220元'),
    ('MX-474折扣10%','MX-474折扣10'),
    ('MX-474余额-5元','MX-474余额5元'),
    ('MX-474余额−5元','MX-474余额5元'),
    ('MX-474余额负五元','MX-474余额五元'),
    ('MX-474选项三还是五？','MX-474选项三还是七？'),
    ('MX-474容量7升','MX-474容量7毫升'),
    ('MX-474重量7公斤','MX-474重量7斤'),
    ('MX-474规格7磅','MX-474规格7两'),
])
def test_normalizer_rejects_changes_to_negative_conditions_or_amounts(raw,candidate):
    q=QueryNormalizer(lambda value:candidate).normalize(raw)
    assert q.canonical==raw
    assert q.downgrade_reason


@pytest.mark.parametrize(('raw','candidate'),[
    ('MX-474Pro我都用了几个月了，保修期还有多久？','MX-474Pro的保修期还有多久？'),
    ('MX-474已经用了几个月，这种情况保修多久？','MX-474已经用了数年，这种情况保修多久？'),
])
def test_normalizer_rejects_dropping_or_changing_vague_duration(raw,candidate):
    q=QueryNormalizer(lambda value:candidate).normalize(raw)
    assert q.canonical==raw
    assert q.downgrade_reason
