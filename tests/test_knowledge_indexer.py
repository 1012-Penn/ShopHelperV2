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
