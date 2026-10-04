"""Opt-in integration against the configured MySQL, synthetic rows cleaned afterward."""

import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import delete, inspect

from app.config import Settings
from app.db.models import Conversation, ConversationSummary, Message
from app.db.session import create_tables, make_engine, make_session_factory
from app.services.context.repository import ContextRepository
from app.services.context.summary import SummaryBatch


@pytest.mark.skipif(
    os.environ.get("CH07_LIVE_MYSQL") != "1",
    reason="opt-in MySQL schema/concurrent append check",
)
def test_mysql_migration_and_duplicate_workers():
    engine = make_engine(Settings.database_url_from_env())
    assert engine.dialect.name == "mysql"
    factory = make_session_factory(engine)
    cid = "ch07-test-" + uuid4().hex
    try:
        create_tables(engine)
        create_tables(engine)
        assert "conversation_summaries" in inspect(engine).get_table_names()
        with factory.begin() as s:
            s.add(Conversation(conversation_id=cid, user_id="ch07-test"))
            s.flush()
            user = Message(conversation_id=cid, role="user", content="订单1001需要核实")
            answer = Message(conversation_id=cid, role="assistant", content="尚未解决")
            s.add_all([user, answer])
            s.flush()
            first, last = user.id, answer.id
        repo = ContextRepository(factory)
        batch = SummaryBatch(cid, None, first, last, "订单1001需要核实", "")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda _: repo.append(batch, "订单1001需核实，尚未解决。"), range(2)
                )
            )
        assert sum(bool(r) for r in results) == 1
        assert len(repo.read(cid)[1]) == 1
        assert repo.read(cid)[0].summary_upto_msg_id == last
    finally:
        with factory.begin() as s:
            s.execute(
                delete(ConversationSummary).where(
                    ConversationSummary.conversation_id == cid
                )
            )
            s.execute(delete(Message).where(Message.conversation_id == cid))
            s.execute(delete(Conversation).where(Conversation.conversation_id == cid))
        engine.dispose()
