from dataclasses import replace

from sqlalchemy import func, select

from app.db.models import FAQ, KnowledgeChunk
from app.services.knowledge.content import ChunkDraft
from app.services.knowledge.indexer import KnowledgeIndexer
from app.services.knowledge.repository import KnowledgeRepository


def _draft(source_key="doc:shipping:0", answer="费用以结算页为准。", **kwargs):
    return ChunkDraft(
        source_key=source_key,
        category="配送",
        questions=["邮费是多少"],
        answer=answer,
        chapter_path=["配送", "运费"],
        content_type="policy",
        is_critical=False,
        **kwargs,
    )


def test_faqs_map_to_knowledge_using_real_question_and_category(db_session_factory):
    with db_session_factory.begin() as session:
        session.add(FAQ(question="邮费是多少？", answer="以结算页为准。", category="配送"))
    repository = KnowledgeRepository(db_session_factory)
    indexer = KnowledgeIndexer(repository)

    ids = indexer.import_faqs()

    with db_session_factory() as session:
        chunk = session.get(KnowledgeChunk, ids[0])
        assert chunk.source_key == "faq:1"
        assert chunk.questions == ["邮费是多少？"]
        assert chunk.category == "配送"
        assert chunk.answer == "以结算页为准。"
        assert chunk.content_type == "product_faq"
        assert chunk.vector_status == "pending"


def test_same_source_same_content_is_idempotent_and_preserves_vector_state(db_session_factory):
    repository = KnowledgeRepository(db_session_factory)
    draft = _draft()
    chunk_id = repository.upsert_drafts([draft])[0]
    repository.mark_vectorized(chunk_id, chunk_id)

    repeated_id = repository.upsert_drafts([draft])[0]

    with db_session_factory() as session:
        assert repeated_id == chunk_id
        assert session.scalar(select(func.count()).select_from(KnowledgeChunk)) == 1
        assert session.get(KnowledgeChunk, chunk_id).vector_status == "vectorized"


def test_changed_content_keeps_primary_key_and_resets_vector_state(db_session_factory):
    repository = KnowledgeRepository(db_session_factory)
    original = _draft()
    chunk_id = repository.upsert_drafts([original])[0]
    repository.mark_vectorized(chunk_id, 901)

    assert repository.upsert_drafts([replace(original, answer="新版运费说明。")]) == [chunk_id]

    pending = repository.pending(limit=10)
    assert [row.id for row in pending] == [chunk_id]
    assert repository.load_by_ids([chunk_id]) == []
    assert pending[0].answer == "新版运费说明。"
    assert pending[0].vector_id is None
    assert pending[0].vector_status == "pending"


def test_import_links_neighbor_chunks_and_pending_is_primary_key_ordered(db_session_factory):
    repository = KnowledgeRepository(db_session_factory)
    drafts = [
        _draft("doc:shipping:0", "第一段。", next_source_key="doc:shipping:1"),
        _draft("doc:shipping:1", "第二段。", previous_source_key="doc:shipping:0"),
    ]

    ids = repository.upsert_drafts(drafts)
    rows = repository.pending(limit=10)

    assert [row.id for row in rows] == ids
    assert rows[0].previous_chunk_id is None
    assert rows[0].next_chunk_id == ids[1]
    assert rows[1].previous_chunk_id == ids[0]
    assert rows[1].next_chunk_id is None


def test_empty_import_is_a_no_op(db_session_factory):
    repository = KnowledgeRepository(db_session_factory)
    assert repository.upsert_drafts([]) == []
    assert KnowledgeIndexer(repository).import_faqs() == []
    assert repository.pending(limit=10) == []


def test_markdown_rebuild_retires_removed_chunks_and_deletes_their_vectors(db_session_factory):
    repository = KnowledgeRepository(db_session_factory)

    class VectorStore:
        def __init__(self):
            self.deleted = []

        def delete(self, ids):
            self.deleted.extend(ids)

    vectors = VectorStore()
    indexer = KnowledgeIndexer(repository, embeddings=object(), vector_store=vectors)
    drafts = [
        _draft("doc:shipping.md:0", "第一条运费政策。", next_source_key="doc:shipping.md:1"),
        _draft("doc:shipping.md:1", "已移除的旧规则。", previous_source_key="doc:shipping.md:0"),
    ]
    old_ids = indexer.import_drafts(drafts)
    for chunk_id in old_ids:
        repository.mark_vectorized(chunk_id, chunk_id)

    indexer.import_markdown([drafts[0]], active_sources=["shipping.md"])

    assert vectors.deleted == [old_ids[1]]
    assert repository.load_by_ids([old_ids[1]]) == []
    assert repository.pending(10) == []
    with db_session_factory() as session:
        retired = session.get(KnowledgeChunk, old_ids[1])
        assert retired.is_active is False and retired.vector_id is None


def test_failed_stale_vector_delete_retries_without_reexposing_removed_content(db_session_factory):
    repository = KnowledgeRepository(db_session_factory)

    class VectorStore:
        def __init__(self):
            self.calls = 0

        def delete(self, ids):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary Milvus outage")

    vectors = VectorStore()
    indexer = KnowledgeIndexer(repository, embeddings=object(), vector_store=vectors)
    drafts = [
        _draft("doc:shipping.md:0", "Current shipping policy."),
        _draft("doc:shipping.md:1", "Removed shipping policy."),
    ]
    old_ids = indexer.import_drafts(drafts)
    for chunk_id in old_ids:
        repository.mark_vectorized(chunk_id, chunk_id)

    try:
        indexer.import_markdown([drafts[0]], active_sources=["shipping.md"])
    except RuntimeError:
        pass
    else:
        assert False, "vector deletion failure must be returned to the CLI"

    assert repository.load_by_ids([old_ids[1]]) == []
    indexer.import_markdown([drafts[0]], active_sources=["shipping.md"])

    assert vectors.calls == 2
    with db_session_factory() as session:
        retired = session.get(KnowledgeChunk, old_ids[1])
        assert retired.is_active is False and retired.vector_id is None


def test_deleted_faq_is_hidden_and_vector_delete_retries(db_session_factory):
    with db_session_factory.begin() as session:
        faq = FAQ(question="邮费是多少？", answer="以结算页为准。", category="配送")
        session.add(faq)
        session.flush()
        faq_id = faq.id

    repository = KnowledgeRepository(db_session_factory)

    class VectorStore:
        def __init__(self):
            self.deleted = []
            self.calls = 0

        def delete(self, ids):
            self.calls += 1
            self.deleted.extend(ids)
            if self.calls == 1:
                raise RuntimeError("temporary Milvus outage")

    vectors = VectorStore()
    indexer = KnowledgeIndexer(repository, embeddings=object(), vector_store=vectors)
    chunk_id = indexer.import_faqs()[0]
    repository.mark_vectorized(chunk_id, chunk_id)
    with db_session_factory.begin() as session:
        session.delete(session.get(FAQ, faq_id))

    try:
        indexer.import_faqs()
    except RuntimeError:
        pass
    else:
        assert False, "vector deletion failure must be returned to the caller"

    assert repository.load_by_ids([chunk_id]) == []
    assert repository.pending(10) == []

    indexer.import_faqs()

    assert vectors.calls == 2
    assert vectors.deleted == [chunk_id, chunk_id]
    with db_session_factory() as session:
        retired = session.get(KnowledgeChunk, chunk_id)
        assert retired.is_active is False and retired.vector_id is None


def test_markdown_faq_content_is_not_reconciled_as_a_legacy_faq_row(db_session_factory):
    repository = KnowledgeRepository(db_session_factory)
    markdown_faq = replace(
        _draft(source_key="doc:product-faq.md:0", answer="商品 FAQ 中的退款说明。"),
        category="商品与订单",
        content_type="product_faq",
    )
    indexer = KnowledgeIndexer(repository)
    chunk_id = indexer.import_drafts([markdown_faq])[0]
    repository.mark_vectorized(chunk_id, chunk_id)

    assert indexer.import_faqs() == []

    assert [row.id for row in repository.load_by_ids([chunk_id])] == [chunk_id]
    with db_session_factory() as session:
        assert session.get(KnowledgeChunk, chunk_id).is_active is True
