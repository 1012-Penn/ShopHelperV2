import pytest
from app.services.knowledge.hybrid_store import HybridStore, HybridRow


class Transport:
    def hybrid_search(self, **kw):
        self.request = kw
        return [[{'id':7,'distance':.8,'entity':{'content_hash':'hash'}}]]
    def search(self, **kw):
        self.request = kw
        return [[{'id':7,'distance':.8,'entity':{'content_hash':'hash'}}]]
    def upsert(self, **kw):
        self.request = kw
        return {'ids':[r['chunk_id'] for r in kw['data']]}


def test_hybrid_filter_both_paths_and_top50():
    client = Transport()
    store = HybridStore(client=client)
    result = store.search([.1]*1024, 'XH-300 续航', 'hybrid', category='耳机" or true')
    assert (result[0].chunk_id, result[0].content_hash) == (7,'hash')
    reqs = client.request['reqs']
    assert all(r.limit == 50 for r in reqs)
    assert reqs[0].expr == reqs[1].expr
    assert '\\"' in reqs[0].expr
    assert all('is_active == true' in r.expr for r in reqs)


def test_hybrid_output_limit_100_keeps_each_search_leg_at_50():
    client=Transport()
    store=HybridStore(client=client)
    store.search([.1]*1024,'XH-300 续航','hybrid_rerank',limit=100)
    assert client.request['limit']==100
    assert [request.limit for request in client.request['reqs']]==[50,50]


def test_upsert_preserves_model_text_and_hash():
    client=Transport()
    store=HybridStore(client=client)
    assert store.upsert([HybridRow(7,[0.]*1024,'XH-300 支持 USB-C','耳机','v1')]) == [7]
    row=client.request['data'][0]
    assert row['text']=='XH-300 支持 USB-C'
    assert row['content_hash']=='v1'
    assert 'bm25' not in row
    with pytest.raises(ValueError):
        store.upsert([HybridRow(7,[0.],'text','耳机','v1')])


def test_bm25_does_not_require_dense_query():
    client=Transport()
    store=HybridStore(client=client)
    assert store.search(None,'型号','bm25')[0].chunk_id == 7
    assert client.request['anns_field']=='bm25'


def test_policy_category_prefixes_use_milvus_like_filter_in_both_search_legs():
    client=Transport()
    store=HybridStore(client=client)
    store.search(
        [.1]*1024,
        '退货申请期限',
        'hybrid_rerank',
        category_prefixes=('退换货与退款','支付与售后'),
    )
    exprs=[request.expr for request in client.request['reqs']]
    assert exprs[0]==exprs[1]
    assert 'category like "退换货与退款%"' in exprs[0]
    assert 'category like "支付与售后%"' in exprs[0]
    assert ' or ' in exprs[0]


def test_allowed_chunk_ids_are_applied_before_hybrid_candidate_truncation():
    client = Transport()
    store = HybridStore(client=client)
    store.search(
        [.1]*1024,
        '退货申请期限',
        'hybrid_rerank',
        category_prefixes=('退换货与退款',),
        chunk_ids=(17, 21),
    )
    exprs=[request.expr for request in client.request['reqs']]
    assert exprs[0] == exprs[1]
    assert 'chunk_id in [17, 21]' in exprs[0]


def test_policy_category_prefix_rejects_empty_and_oversized_values():
    client=Transport()
    store=HybridStore(client=client)
    with pytest.raises(ValueError):
        store.search([.1]*1024,'query','hybrid',category_prefixes=(' ',))
    with pytest.raises(ValueError):
        store.search([.1]*1024,'query','hybrid',category_prefixes=('超'*200,))
