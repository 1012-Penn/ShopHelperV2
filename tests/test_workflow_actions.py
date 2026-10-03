import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import func, select, text, inspect

from app.db.models import Message, Ticket
from app.db.session import make_engine, make_session_factory, create_tables
from app.schemas import ChatRequest
from app.tools.registry import ToolRegistry, ToolRunner, ToolResult


@pytest.fixture
def factory(tmp_path):
    engine = make_engine(f'sqlite:///{tmp_path}/actions.db')
    create_tables(engine)
    yield make_session_factory(engine)
    engine.dispose()


def setup(factory, actions=('handoff', 'create_ticket')):
    from app.services.workflow.storage import ConversationStore, ConversationLocks
    from app.services.workflow.actions import TicketActions
    store = ConversationStore(factory)
    store.prepare(ChatRequest(conversation_id='c', message='我要投诉'))
    message_id = store.save_answer('c', '很抱歉', [], list(actions), '我要投诉')
    runner_factory = lambda tools: ToolRunner(ToolRegistry(tools), 1, 0)
    return TicketActions(factory, runner_factory, ConversationLocks()), message_id


def test_suggestions_alone_have_no_ticket_side_effect(factory):
    setup(factory)
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(Ticket)) == 0


def test_confirmed_button_calls_existing_tool_and_duplicate_returns_same_ticket(factory):
    service, mid = setup(factory)
    first = service.submit('c', 'demo-user', mid)
    duplicate = service.submit('c', 'demo-user', mid)
    assert first == duplicate
    assert first['status'] == 'created'
    with factory() as s:
        tickets = list(s.scalars(select(Ticket)))
        assert len(tickets) == 1
        assert tickets[0].description == '我要投诉'
        assert tickets[0].conversation_id == 'c'


def test_concurrent_button_requests_create_only_one_ticket(factory):
    service, mid = setup(factory)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: service.submit('c', 'demo-user', mid), range(2)))
    assert results[0] == results[1]
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(Ticket)) == 1


@pytest.mark.parametrize('who,actions', [('other-user', ['create_ticket']), ('demo-user', ['handoff'])])
def test_unauthorized_or_unsuggested_action_rejected(factory, who, actions):
    service, mid = setup(factory, actions)
    with pytest.raises(ValueError):
        service.submit('c', who, mid)
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(Ticket)) == 0


@pytest.mark.parametrize('status', ['submitting', 'unknown'])
def test_pending_or_unknown_is_not_reexecuted_after_service_restart(factory, status):
    service, mid = setup(factory)
    with factory.begin() as s:
        row = s.get(Message, mid)
        row.actions = {**row.actions, 'ticket': {'status': status}}
    service.runner_factory = lambda tools: pytest.fail('must not retry write')
    result = service.submit('c', 'demo-user', mid)
    assert result['status'] == status
    assert 'ticket_no' not in result


def test_tool_failure_records_unknown_and_never_claims_created(factory):
    service, mid = setup(factory)
    class Failure:
        def run(self, *args):
            return ToolResult('create_ticket', 'x', '结果暂未确认', True)
        def close(self):
            pass
    service.runner_factory = lambda tools: Failure()
    assert service.submit('c', 'demo-user', mid)['status'] == 'unknown'
    service.runner_factory = lambda tools: pytest.fail('must not retry')
    assert service.submit('c', 'demo-user', mid)['status'] == 'unknown'


def test_migration_is_repeatable_and_preserves_legacy_messages(tmp_path):
    from scripts.migrate_ch05 import migrate_ch05
    engine = make_engine(f'sqlite:///{tmp_path}/legacy.db')
    with engine.begin() as c:
        c.execute(text('CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT)'))
        c.execute(text("INSERT INTO messages VALUES (1, '旧消息')"))
    migrate_ch05(engine)
    migrate_ch05(engine)
    assert 'actions' in {col['name'] for col in inspect(engine).get_columns('messages')}
    with engine.connect() as c:
        assert c.execute(text('SELECT content FROM messages')).scalar() == '旧消息'
    engine.dispose()


def test_store_rejects_conversation_owner_change(factory):
    from app.services.workflow.storage import ConversationStore
    store = ConversationStore(factory)
    store.prepare(ChatRequest(conversation_id='c', message='你好', user_id='owner'))
    with pytest.raises(ValueError):
        store.prepare(ChatRequest(conversation_id='c', message='你好', user_id='intruder'))


def test_ticket_runner_does_not_retry_unknown_write_failure():
    from langchain.tools import tool
    executions = []
    @tool('create_ticket')
    def bad_write(description: str, ticket_type: str) -> str:
        """A write whose result cannot be confirmed."""
        executions.append(description)
        raise RuntimeError('connection lost after commit')
    runner = ToolRunner(ToolRegistry([bad_write]), 1, 2)
    result = runner.run('create_ticket', {'description': '投诉', 'ticket_type': '投诉'}, 'id')
    assert result.is_error
    assert executions == ['投诉']
