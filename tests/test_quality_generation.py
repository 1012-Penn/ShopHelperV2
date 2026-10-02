from app.services.quality.generation import arrange_evidence, KnowledgeAnswerService
from app.services.quality.retrieval import Evidence
from app.services.quality.ledger import QualityLedger
from app.db.models import Conversation, LowConfidenceQuestion
from sqlalchemy import select
import pytest


def evidence(n=1):
    return Evidence(n,n,['售后','退款'],'到账时间','不承诺到账时间，以银行为准。','doc:refund:0',.9)


def test_prompt_order_keeps_numbers():
    assert [e.n for e in arrange_evidence([evidence(i) for i in range(1,7)])]==[1,3,5,6,4,2]


def test_insufficient_and_invalid_citations_are_refused_and_pooled(db_session_factory):
    class Retriever:
        def retrieve(self,*args,**kwargs):
            return [evidence()]
    with db_session_factory.begin() as s:
        s.add(Conversation(conversation_id='c',user_id='u'))
    ledger=QualityLedger(db_session_factory)
    service=KnowledgeAnswerService(Retriever(),lambda q,ev:dict(sufficient=False,reason='没有证据',answer='编造',cited_numbers=[]),ledger)
    result=service.answer('明天肯定到账吗','c')
    assert result.refused and '编造' not in result.answer
    service.generator=lambda q,ev:dict(sufficient=True,reason='',answer='以银行为准[99]',cited_numbers=[99])
    assert service.answer('到账','c').refused
    with db_session_factory() as s:
        rows=list(s.scalars(select(LowConfidenceQuestion)))
        assert len(rows)==2
        assert rows[0].raw_question=='明天肯定到账吗' and rows[0].source=='self_check'


def test_absent_retrieval_and_pool_failure_do_not_succeed():
    class Empty:
        def retrieve(self,*a,**kw):return []
    class FailingLedger:
        def add_low_confidence(self,*a,**kw):raise RuntimeError('DB down')
    service=KnowledgeAnswerService(Empty(),None,FailingLedger())
    with pytest.raises(RuntimeError):service.answer('未知问题','c')
    assert service.generate('未知问题',[]).source=='retrieval_low_conf'


def test_answer_records_full_prompt_evidence():
    service=KnowledgeAnswerService(None,lambda q,ev:dict(sufficient=True,reason='',answer='以银行为准[1]',cited_numbers=[1]),None)
    result=service.generate('到账时间',[evidence(),evidence(2)])
    assert not result.refused
    assert [c['n'] for c in result.citations]==[1,2]


def test_no_tool_policy_cannot_bypass_gate(db_session_factory):
    from app.services.chat import ChatService
    from app.schemas import ChatRequest
    from langchain_core.messages import AIMessageChunk
    class Model:
        def bind_tools(self,tools):return self
        def stream(self,messages):yield AIMessageChunk(content='明天肯定到账')
    class Runner:
        class registry:
            tools=[]
    class Service:
        def answer(self,*a,**kw):
            from app.services.quality.generation import GuardedAnswer, REFUSAL
            return GuardedAnswer(REFUSAL,[],True,'证据不足','self_check')
    chat=ChatService(db_session_factory,lambda:Model(),lambda tools:Runner(),knowledge_answer_service=Service())
    events=list(chat.stream_events(ChatRequest(conversation_id='guard',message='退款明天肯定到账吗')))
    text=''.join(e['data'].get('content','') for e in events)
    assert '肯定到账' not in text and '无法确认' in text
    assert events[-1]['event']=='done'


def test_invalid_structured_output_is_self_check_refusal():
    service=KnowledgeAnswerService(None,lambda q,e:{'answer':'乱写'},None)
    result=service.generate('退款',[evidence()])
    assert result.refused and result.source=='self_check'


def test_failed_knowledge_tool_has_paired_error_result(db_session_factory):
    from app.services.chat import ChatService
    from app.schemas import ChatRequest
    from app.db.models import Message
    from langchain_core.messages import AIMessageChunk
    class Model:
        def bind_tools(self,t):return self
        def stream(self,m):yield AIMessageChunk(content='',tool_calls=[{'id':'tc','name':'query_faq','args':{'query':'退款'}}])
    class Runner:
        class registry:tools=[]
    class Failing:
        def answer(self,*a,**kw):raise RuntimeError('provider failure')
    chat=ChatService(db_session_factory,lambda:Model(),lambda t:Runner(),knowledge_answer_service=Failing())
    assert list(chat.stream_events(ChatRequest(conversation_id='fail',message='退款')))[-1]['event']=='error'
    with db_session_factory() as s:
        messages=list(s.scalars(select(Message)))
        assert any(m.role=='tool' and m.tool_call_id=='tc' for m in messages)


def test_order_clarification_preserved(db_session_factory):
    from app.services.chat import ChatService
    from app.schemas import ChatRequest
    from langchain_core.messages import AIMessageChunk
    class Model:
        def bind_tools(self,t):return self
        def stream(self,m):yield AIMessageChunk(content='请提供订单号，我帮您查询。')
    class Runner:
        class registry:tools=[]
    class FailIfUsed:
        def answer(self,*a,**kw):raise RuntimeError('should not use knowledge')
    chat=ChatService(db_session_factory,lambda:Model(),lambda t:Runner(),knowledge_answer_service=FailIfUsed())
    events=list(chat.stream_events(ChatRequest(conversation_id='clarify',message='帮我查一下订单')))
    assert '请提供订单号' in ''.join(e['data'].get('content','') for e in events)
    assert events[-1]['event']=='done'
