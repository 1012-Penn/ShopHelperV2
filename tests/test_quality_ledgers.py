import pytest
from sqlalchemy import select, text
from app.db.models import Conversation, Message
from app.services.quality.ledger import QualityLedger
from scripts.migrate_ch04 import migrate_ch04


def test_conversation_identity_and_low_confidence(db_session_factory):
    with db_session_factory.begin() as s:
        s.add_all([Conversation(conversation_id='a', user_id='u'), Conversation(conversation_id='b', user_id='u')])
    with db_session_factory() as s:
        rows = list(s.scalars(select(Conversation).order_by(Conversation.conversation_id)))
        assert all(isinstance(r.id, int) and r.id > 0 for r in rows)
        assert rows[0].id != rows[1].id
        numeric_id = rows[0].id
    ledger = QualityLedger(db_session_factory)
    ledger.add_low_confidence('a', '到底啥时候到账啊', 'self_check', '证据不足')
    with db_session_factory() as s:
        from app.db.models import LowConfidenceQuestion
        row = s.scalar(select(LowConfidenceQuestion))
        assert (row.conversation_id, row.raw_question, row.source) == (numeric_id, '到底啥时候到账啊', 'self_check')
    with pytest.raises(ValueError):
        ledger.add_low_confidence('missing', '原话', 'self_check', '不足')
    with pytest.raises(ValueError):
        ledger.add_low_confidence('a', '原话', 'bad', '不足')


def test_recurrence_and_required_resolution(db_session_factory):
    ledger = QualityLedger(db_session_factory)
    case = dict(eval_id='A01', bucket='A_policy', query='何时到账', strategy='hybrid_rerank', answer='立即到账[1]', reason='证据未承诺', citations=[{'n': 1, 'answer': '以银行为准'}], judge_model='judge')
    ledger.record_faith_case(case)
    with pytest.raises(ValueError):
        ledger.resolve('A01', '已解决', ' ')
    ledger.resolve('A01', '已解决', '修正文案')
    ledger.record_faith_case({**case, 'answer': '次日到账[1]'})
    row = ledger.list_cases()[0]
    assert (row.seen_count, row.status, row.resolution) == (2, '未解决', None)
    assert row.resolved_at is not None
    assert row.answer == '次日到账[1]'
    assert row.citations == case['citations']


def test_migration_twice_and_existing_conversation():
    from app.db.session import make_engine, create_tables
    engine = make_engine('sqlite:///:memory:')
    with engine.begin() as c:
        c.execute(text('CREATE TABLE conversations (conversation_id VARCHAR(128) PRIMARY KEY, user_id VARCHAR(128), status VARCHAR(32), created_at DATETIME)'))
        c.execute(text("INSERT INTO conversations VALUES ('old','u','open',CURRENT_TIMESTAMP)"))
        c.execute(text('CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id VARCHAR(128), role VARCHAR(16), content TEXT, tool_calls JSON, tool_call_id VARCHAR(128), created_at DATETIME)'))
    migrate_ch04(engine)
    create_tables(engine)
    migrate_ch04(engine)
    with engine.connect() as c:
        assert c.execute(text("SELECT id FROM conversations WHERE conversation_id='old'")).scalar_one() > 0
        c.execute(text("INSERT INTO conversations (conversation_id,user_id) VALUES ('new','u')"))
        assert c.execute(text("SELECT id FROM conversations WHERE conversation_id='new'")).scalar_one() > 0


def test_answer_evidence_persists(db_session_factory):
    with db_session_factory.begin() as s:
        s.add(Conversation(conversation_id='c', user_id='u'))
        s.flush()
        s.add(Message(conversation_id='c', role='assistant', content='答[1]', citations=[{'n':1,'chunk_id':7}]))
    with db_session_factory() as s:
        assert s.scalar(select(Message)).citations == [{'n':1,'chunk_id':7}]
