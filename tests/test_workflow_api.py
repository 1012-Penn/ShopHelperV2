import json

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db.models import Ticket
from app.config import Settings
from app.main import create_app
from workflow_helpers import ScriptedModel, make_workflow


def test_tickets_api_requires_suggestion_and_correct_owner(tmp_path):
    workflow = make_workflow(tmp_path,ScriptedModel('投诉'))
    with TestClient(create_app(chat_service=workflow)) as client:
        response = client.post('/api/v1/chat/stream',json={'conversation_id':'api','message':'我要投诉'})
        assert response.status_code==200
        done = next(json.loads(block.split('data: ',1)[1]) for block in response.text.split('\n\n')
                    if block.startswith('event: done'))
        request = {'conversation_id':'api','user_id':'demo-user','message_id':done['message_id']}
        denied = client.post('/api/v1/tickets',json={**request,'user_id':'intruder'})
        assert denied.status_code==403
        first = client.post('/api/v1/tickets',json=request)
        again = client.post('/api/v1/tickets',json=request)
        assert first.status_code==200
        assert first.json()==again.json()
        assert first.json()['status']=='created'
        with workflow.session_factory() as s:
            assert s.scalar(select(func.count()).select_from(Ticket))==1


def test_lifespan_closes_checkpoint_after_last_request(tmp_path):
    workflow = make_workflow(tmp_path)
    with TestClient(create_app(chat_service=workflow)) as client:
        assert client.get('/health').status_code==200
    import sqlite3
    try:
        workflow.checkpointer.conn.execute('SELECT 1')
    except sqlite3.ProgrammingError:
        return
    assert False, 'checkpoint connection must be closed'


def test_default_factory_greeting_does_not_initialize_knowledge_or_model(tmp_path,monkeypatch):
    import app.services.workflow.runtime as runtime
    settings = Settings('test','unused','https://provider.invalid',f'sqlite:///{tmp_path}/runtime.db')
    monkeypatch.setattr(runtime,'ChatOpenAI',lambda **kwargs: (_ for _ in ()).throw(AssertionError('no model')))
    workflow = runtime.build_workflow_service(settings,{
        'WORKFLOW_CHECKPOINT_PATH':str(tmp_path/'saver.sqlite'),
        'WORKFLOW_LOG_PATH':str(tmp_path/'log.jsonl'), 'QUALITY_ENABLED':'true'})
    with TestClient(create_app(chat_service=workflow)) as client:
        response = client.post('/api/v1/chat/stream',json={'conversation_id':'g','message':'你好'})
        assert 'event: done' in response.text
        assert 'event: error' not in response.text


def test_blank_message_and_invalid_ticket_id_rejected(tmp_path):
    workflow = make_workflow(tmp_path)
    with TestClient(create_app(chat_service=workflow)) as client:
        assert client.post('/api/v1/chat/stream',json={'conversation_id':'c','message':'   '}).status_code==422
        assert client.post('/api/v1/tickets',json={'conversation_id':'c','message_id':0}).status_code==422
