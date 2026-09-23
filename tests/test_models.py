from sqlalchemy import func, select

from app.db.models import Conversation, FAQ, Message, Ticket
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
    from sqlalchemy.dialects.mysql import dialect
    from sqlalchemy.schema import CreateTable

    ddl = str(CreateTable(FAQ.__table__).compile(dialect=dialect()))
    assert "UNIQUE (question)" not in ddl
