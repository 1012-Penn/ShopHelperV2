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
