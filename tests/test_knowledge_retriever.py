from types import SimpleNamespace

from app.services.knowledge.retriever import DenseSearcher, FAQHit, KnowledgeRetriever
from app.services.knowledge.vector_store import VectorHit


class FakeEmbeddings:
    def __init__(self):
        self.queries = []

    def embed_query(self, query):
        self.queries.append(query)
        return [0.1] * 1024


class FakeVectorStore:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, vector, limit):
        self.calls.append((vector, limit))
        return self.hits[:limit]


def test_dense_searcher_embeds_query_then_calls_milvus_top_k():
    embeddings = FakeEmbeddings()
    vectors = FakeVectorStore([VectorHit(10, 0.88)])
    searcher = DenseSearcher(embeddings, vectors)

    result = searcher.search("邮费是多少？", limit=5)

    assert embeddings.queries == ["邮费是多少？"]
    assert vectors.calls == [([0.1] * 1024, 5)]
    assert result == [VectorHit(10, 0.88)]


def test_retriever_filters_threshold_hydrates_in_vector_order_and_caps_at_five():
    hits = [
        VectorHit(20, 0.91),
        VectorHit(99, 0.88),  # Dangling Milvus primary key has no authoritative MySQL row.
        VectorHit(10, 0.39),  # Below the configured threshold.
        VectorHit(30, 0.81),
    ]
    searcher = FakeVectorStore(hits)
    rows = {
        20: SimpleNamespace(id=20, questions=["运费是多少？"], category="配送", answer="看结算页。"),
        30: SimpleNamespace(id=30, questions=["如何申请退货？"], category="退换货", answer="从订单申请。"),
    }

    class Repository:
        def load_by_ids(self, ids):
            return [rows[item] for item in ids if item in rows]

    retriever = KnowledgeRetriever(searcher, Repository(), top_k=12, min_similarity=0.4)

    result = retriever.search("邮费是多少？")

    assert searcher.calls[0][1] == 5
    assert result == [
        FAQHit("运费是多少？", "看结算页。", "配送", 0.91),
        FAQHit("如何申请退货？", "从订单申请。", "退换货", 0.81),
    ]


def test_retriever_uses_chapter_heading_when_policy_has_no_questions():
    class Searcher:
        def search(self, query, limit):
            return [VectorHit(1, 0.7)]

    class Repository:
        def load_by_ids(self, ids):
            return [SimpleNamespace(id=1, questions=[], chapter_path=["配送", "运费计算"], category="配送", answer="以结算页为准。")]

    result = KnowledgeRetriever(Searcher(), Repository(), top_k=5, min_similarity=0.4).search("运费怎么计算")

    assert result == [FAQHit("运费计算", "以结算页为准。", "配送", 0.7)]


def test_retriever_returns_no_hits_for_empty_or_below_threshold_results():
    class Searcher:
        def search(self, query, limit):
            return [VectorHit(1, 0.2)]

    class Repository:
        def load_by_ids(self, ids):
            raise AssertionError("below-threshold IDs should not be hydrated")

    retriever = KnowledgeRetriever(Searcher(), Repository(), top_k=5, min_similarity=0.4)

    assert retriever.search("unrelated") == []
