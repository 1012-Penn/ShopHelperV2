from types import SimpleNamespace

import pytest

from app.services.knowledge.content import ChunkDraft
from app.services.knowledge.embeddings import EmbeddingClient
from app.services.knowledge.indexer import KnowledgeIndexer
from app.services.knowledge.repository import KnowledgeRepository
from app.services.knowledge.vector_store import MilvusKnowledgeStore, VectorHit, VectorRow


class FakeEmbeddingAPI:
    def __init__(self, data=None, error=None):
        self.data = data or []
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(data=self.data)


class FakeOpenAI:
    def __init__(self, data=None, error=None):
        self.embeddings = SimpleNamespace(create=FakeEmbeddingAPI(data, error).create)
        self.api_key = ""
        self.base_url = ""


def test_embedding_response_is_reordered_by_input_index_and_validates_dimension():
    client = FakeOpenAI(
        [
            SimpleNamespace(index=1, embedding=[2.0] * 1024),
            SimpleNamespace(index=0, embedding=[1.0] * 1024),
        ]
    )
    embeddings = EmbeddingClient(api_key="test-key", base_url="https://embed.example/v1", client=client)

    vectors = embeddings.embed_documents(["first", "second"])

    assert vectors == [[1.0] * 1024, [2.0] * 1024]
    assert client.embeddings.create.__self__.calls[0] == {
        "model": "BAAI/bge-m3",
        "input": ["first", "second"],
    }


def test_embedding_errors_do_not_expose_provider_exception_text():
    embeddings = EmbeddingClient(
        api_key="secret-key", base_url="https://embed.example/v1", client=FakeOpenAI(error=RuntimeError("secret-key"))
    )

    with pytest.raises(RuntimeError) as error:
        embeddings.embed_query("邮费是多少")

    assert "secret-key" not in str(error.value)


def test_embedding_rejects_unexpected_vector_dimensions():
    embeddings = EmbeddingClient(
        api_key="test-key",
        base_url="https://embed.example/v1",
        client=FakeOpenAI([SimpleNamespace(index=0, embedding=[1.0] * 3)]),
    )
    with pytest.raises(ValueError, match="1024"):
        embeddings.embed_query("question")


class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, *args, **kwargs):
        self.fields.append((args, kwargs))


class FakeMilvusClient:
    def __init__(self):
        self.exists = False
        self.schema = None
        self.collection_kwargs = None
        self.upserts = []
        self.fail_upsert_times = 0
        self.loaded_collection = None

    def has_collection(self, collection_name):
        return self.exists

    def create_schema(self, **kwargs):
        self.schema = FakeSchema()
        self.schema.auto_id = kwargs.get("auto_id")
        return self.schema

    def create_collection(self, **kwargs):
        self.collection_kwargs = kwargs
        self.exists = True

    def load_collection(self, collection_name):
        self.loaded_collection = collection_name

    def prepare_index_params(self):
        return FakeIndexParams()

    def upsert(self, collection_name, data):
        if self.fail_upsert_times:
            self.fail_upsert_times -= 1
            raise RuntimeError("simulated Milvus interruption")
        self.upserts.append((collection_name, list(data)))
        return {"ids": [row["chunk_id"] for row in data]}

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return [[{"id": 12, "distance": 0.82}, {"id": 8, "distance": 0.71}]]

    def delete(self, **kwargs):
        self.deleted = kwargs


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, *args, **kwargs):
        self.indexes.append((args, kwargs))


def test_milvus_collection_uses_mysql_integer_primary_key_and_cosine():
    from pymilvus import DataType

    client = FakeMilvusClient()
    store = MilvusKnowledgeStore(client=client, collection_name="knowledge", dimension=1024)

    store.ensure_collection()

    assert client.collection_kwargs["schema"].fields == [
        (("chunk_id", DataType.INT64), {"is_primary": True}),
        (("embedding", DataType.FLOAT_VECTOR), {"dim": 1024}),
    ]
    assert client.collection_kwargs["index_params"].indexes[0][1]["metric_type"] == "COSINE"
    assert client.collection_kwargs["schema"].auto_id is False
    assert client.loaded_collection == "knowledge"


def test_milvus_upsert_uses_known_chunk_ids_and_search_parses_cosine_hits():
    client = FakeMilvusClient()
    store = MilvusKnowledgeStore(client=client, collection_name="knowledge", dimension=1024)

    ids = store.upsert([VectorRow(12, [0.1] * 1024)])
    hits = store.search([0.2] * 1024, limit=2)

    assert ids == [12]
    assert client.upserts[0][1] == [{"chunk_id": 12, "embedding": [0.1] * 1024}]
    assert hits == [VectorHit(12, 0.82), VectorHit(8, 0.71)]
    assert client.search_kwargs["search_params"]["metric_type"] == "COSINE"
    store.delete([12])
    assert client.deleted["ids"] == [12]


def _repository(db_session_factory):
    repository = KnowledgeRepository(db_session_factory)
    chunk_id = repository.upsert_drafts(
        [
            ChunkDraft(
                source_key="doc:shipping:0",
                category="配送",
                questions=["邮费是多少"],
                answer="以结算页为准。",
                chapter_path=["配送", "运费"],
                content_type="policy",
                is_critical=False,
            )
        ]
    )[0]
    return repository, chunk_id


class StaticEmbeddings:
    def embed_documents(self, texts):
        return [[float(index + 1)] * 1024 for index, _ in enumerate(texts)]

    def embed_query(self, text):
        return [1.0] * 1024


def test_sync_marks_mysql_vectorized_only_after_successful_milvus_upsert(db_session_factory):
    repository, chunk_id = _repository(db_session_factory)
    client = FakeMilvusClient()
    indexer = KnowledgeIndexer(
        repository, StaticEmbeddings(), MilvusKnowledgeStore(client=client, collection_name="knowledge")
    )

    summary = indexer.sync_pending(batch_size=8)

    row = repository.load_by_ids([chunk_id])[0]
    assert summary.pending_before == 1 and summary.vectorized == 1 and summary.failed == 0
    assert row.vector_id == chunk_id and row.vector_status == "vectorized"


def test_retry_after_milvus_success_before_mysql_backfill_upserts_same_primary_key(db_session_factory, monkeypatch):
    repository, chunk_id = _repository(db_session_factory)
    client = FakeMilvusClient()
    store = MilvusKnowledgeStore(client=client, collection_name="knowledge")
    indexer = KnowledgeIndexer(repository, StaticEmbeddings(), store)
    original_mark = repository.mark_vectorized
    fail_once = True

    def fail_before_backfill(current_id, vector_id):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("simulated interruption")
        original_mark(current_id, vector_id)

    monkeypatch.setattr(repository, "mark_vectorized", fail_before_backfill)
    first = indexer.sync_pending(batch_size=8)
    assert first.failed == 1 and repository.pending(limit=8)[0].id == chunk_id

    second = indexer.sync_pending(batch_size=8)

    assert second.vectorized == 1
    assert [row["chunk_id"] for _, rows in client.upserts for row in rows] == [chunk_id, chunk_id]
    row = repository.load_by_ids([chunk_id])[0]
    assert row.vector_id == chunk_id and row.vector_status == "vectorized"


def test_milvus_failure_leaves_mysql_pending_for_retry(db_session_factory):
    repository, chunk_id = _repository(db_session_factory)
    client = FakeMilvusClient()
    client.fail_upsert_times = 2  # batch attempt plus per-row isolation retry
    indexer = KnowledgeIndexer(
        repository, StaticEmbeddings(), MilvusKnowledgeStore(client=client, collection_name="knowledge")
    )

    first = indexer.sync_pending(batch_size=8)
    assert first.failed == 1 and repository.pending(limit=8)[0].id == chunk_id
    second = indexer.sync_pending(batch_size=8)
    assert second.vectorized == 1
