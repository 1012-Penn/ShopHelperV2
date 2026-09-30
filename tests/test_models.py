from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.mysql import dialect
from sqlalchemy.schema import CreateTable
import pytest

from app.db.models import (
    Conversation,
    FAQ,
    KnowledgeChunk,
    KnowledgeExtractionCursor,
    KnowledgeQAStaging,
    Message,
    Ticket,
)
from scripts.seed import seed_faq


def test_seed_is_idempotent_and_models_store_tool_calls(db_session):
    seed_faq(db_session)
    seed_faq(db_session)
    assert db_session.scalar(select(func.count()).select_from(FAQ)) == 4

    db_session.add(Conversation(conversation_id="demo-1", user_id="demo-user", status="open"))
    db_session.flush()
    message = Message(conversation_id="demo-1", role="assistant", content="", tool_calls=[{"id": "call-1"}])
    db_session.add(message)
    db_session.commit()

    saved = db_session.get(Message, message.id)
    assert saved.tool_calls[0]["id"] == "call-1"
    assert saved.conversation.user_id == "demo-user"


def test_ticket_is_linked_to_conversation(db_session):
    db_session.add(Conversation(conversation_id="ticket-conv", user_id="demo-user", status="open"))
    db_session.flush()
    db_session.add(Ticket(ticket_no="T1001", conversation_id="ticket-conv", description="商品故障", ticket_type="退货", status="open"))
    db_session.commit()
    assert db_session.get(Ticket, "T1001").conversation.conversation_id == "ticket-conv"


def test_mysql_faq_ddl_avoids_unique_index_on_text_question():
    ddl = str(CreateTable(FAQ.__table__).compile(dialect=dialect()))
    assert "UNIQUE (question)" not in ddl


def test_knowledge_models_store_json_fields_and_neighbor_pointers(db_session):
    left = KnowledgeChunk(
        source_key="doc:shipping:0",
        category="配送",
        questions=["运费是多少"],
        answer="费用以结算页为准。",
        embedding_text="配送\n运费是多少\n费用以结算页为准。",
        chapter_path=["配送", "运费"],
        content_type="policy",
        is_critical=False,
        vector_status="pending",
        content_hash="hash-left",
    )
    right = KnowledgeChunk(
        source_key="doc:shipping:1",
        category="配送",
        questions=["如何查看运费"],
        answer="提交订单前可查看费用。",
        embedding_text="配送\n如何查看运费\n提交订单前可查看费用。",
        chapter_path=["配送", "运费"],
        content_type="policy",
        is_critical=False,
        vector_status="pending",
        content_hash="hash-right",
    )
    db_session.add_all([left, right])
    db_session.flush()
    left.next_chunk_id = right.id
    right.previous_chunk_id = left.id
    db_session.flush()

    saved = db_session.get(KnowledgeChunk, left.id)
    assert saved.questions == ["运费是多少"]
    assert saved.chapter_path == ["配送", "运费"]
    assert saved.next_chunk_id == right.id
    assert db_session.get(KnowledgeChunk, right.id).previous_chunk_id == left.id
    assert saved.vector_id is None
    assert saved.vector_status == "pending"


def test_knowledge_staging_and_cursor_tables_are_created(db_session_factory):
    from sqlalchemy import inspect

    names = set(inspect(db_session_factory.kw["bind"]).get_table_names())
    assert {"knowledge_chunks", "knowledge_qa_staging", "knowledge_extraction_cursor"} <= names


def test_staging_unique_guard_prevents_reprocessing_same_source_pair(db_session):
    conversation = Conversation(conversation_id="knowledge-source", user_id="demo", status="open")
    db_session.add(conversation)
    db_session.flush()
    user_message = Message(conversation_id=conversation.conversation_id, role="user", content="邮费是多少")
    answer_message = Message(conversation_id=conversation.conversation_id, role="assistant", content="看结算页")
    db_session.add_all([user_message, answer_message])
    db_session.flush()
    candidate = KnowledgeQAStaging(
        source_user_message_id=user_message.id,
        source_assistant_message_id=answer_message.id,
        category="配送",
        questions=["邮费是多少"],
        answer="费用以结算页为准。",
        fingerprint="canonical-hash",
        run_id="run-1",
    )
    db_session.add(candidate)
    db_session.commit()

    db_session.add(
        KnowledgeQAStaging(
            source_user_message_id=user_message.id,
            source_assistant_message_id=answer_message.id,
            category="配送",
            questions=["邮费是多少"],
            answer="费用以结算页为准。",
            fingerprint="canonical-hash",
            run_id="run-2",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_mysql_knowledge_ddl_uses_bounded_unique_keys():
    chunk_ddl = str(CreateTable(KnowledgeChunk.__table__).compile(dialect=dialect()))
    stage_ddl = str(CreateTable(KnowledgeQAStaging.__table__).compile(dialect=dialect()))
    assert "UNIQUE (source_key)" in chunk_ddl
    assert "UNIQUE (fingerprint, source_user_message_id)" in stage_ddl
    assert "UNIQUE (answer)" not in chunk_ddl


def test_knowledge_vector_status_rejects_unknown_values(db_session):
    db_session.add(
        KnowledgeChunk(
            source_key="doc:invalid-status:0",
            category="配送",
            questions=["问题"],
            answer="答案。",
            embedding_text="配送\n问题\n答案。",
            chapter_path=["配送"],
            content_type="policy",
            is_critical=False,
            vector_status="processing",
            content_hash="invalid-status-hash",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
