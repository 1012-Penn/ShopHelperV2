"""Network-boundary fixtures; the graph, persistence, tools and ledger stay real."""
import json
import sqlite3

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.sqlite import SqliteSaver

from app.db.session import make_engine, make_session_factory, create_tables
from app.tools.registry import ToolRegistry, ToolRunner


class ScriptedModel:
    def __init__(self, intent='物流', decisions=None, chunks=None):
        self.intent = intent
        self.decisions = list(decisions or [])
        self.chunks = chunks or ['模拟物流', '运输中']
        self.calls = []
        self.bound_tools = []

    def bind(self, **kwargs):
        self.binding = kwargs
        return self

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if '七类' in messages[0].content:
            return AIMessage(content=json.dumps({'intent': self.intent}, ensure_ascii=False))
        if self.decisions:
            return self.decisions.pop(0)
        return AIMessage(content='{"next":"answer","actions":[]}')

    def stream(self, messages):
        self.calls.append(messages)
        for part in self.chunks:
            yield AIMessageChunk(content=part)


class EvidenceRetriever:
    def __init__(self, evidence=None, error=None):
        self.evidence = [] if evidence is None else evidence
        self.error = error
        self.queries = []

    def retrieve(self, question, category=None):
        self.queries.append((question, category))
        if self.error:
            raise self.error
        return self.evidence, {'strategy': 'hybrid_rerank'}


def make_workflow(tmp_path, model=None, retriever=None, limits=None):
    from app.services.workflow.graph import WorkflowService
    engine = make_engine(f'sqlite:///{tmp_path}/business.db')
    create_tables(engine)
    factory = make_session_factory(engine)
    saver = SqliteSaver(sqlite3.connect(str(tmp_path / 'checkpoint.db'), check_same_thread=False))
    service = WorkflowService(factory, lambda: model or ScriptedModel(),
                              lambda tools: ToolRunner(ToolRegistry(tools), 1, 0),
                              retriever or EvidenceRetriever(), saver,
                              tmp_path / 'workflow.jsonl', **({'limits': limits} if limits else {}))
    return service
