"""Network-boundary fixtures; the graph, persistence, tools and ledger stay real."""

import json
import sqlite3

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.sqlite import SqliteSaver

from app.db.session import create_tables, make_engine, make_session_factory
from app.tools.registry import ToolRegistry, ToolRunner


class ScriptedModel:
    def __init__(self, intent="物流", decisions=None, chunks=None, confidence=0.92):
        self.intent = intent
        self.confidence = confidence
        self.decisions = list(decisions or [])
        self.chunks = chunks or ["模拟物流", "运输中"]
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
        if "本轮问题独立化节点" in messages[0].content:
            question = messages[-1].content.split("本轮原问题：\n", 1)[-1]
            return AIMessage(
                content=json.dumps(
                    {"question": question, "reference_resolved": True},
                    ensure_ascii=False,
                )
            )
        if "本轮意图选择节点" in messages[0].content:
            return AIMessage(
                content=json.dumps(
                    {"intent": self.intent, "confidence": self.confidence},
                    ensure_ascii=False,
                )
            )
        if "检索查询扩写节点" in messages[0].content:
            return AIMessage(content=json.dumps({"queries": [messages[-1].content]}))
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

    def retrieve(self, question, category=None, **kwargs):
        self.queries.append((question, category) if not kwargs else (question, category, kwargs))
        if self.error:
            raise self.error
        return self.evidence, {"strategy": "hybrid_rerank"}


def make_workflow(tmp_path, model=None, retriever=None, limits=None):
    from app.services.workflow.graph import WorkflowService

    engine = make_engine(f"sqlite:///{tmp_path}/business.db")
    create_tables(engine)
    factory = make_session_factory(engine)
    saver = SqliteSaver(
        sqlite3.connect(str(tmp_path / "checkpoint.db"), check_same_thread=False)
    )
    service = WorkflowService(
        factory,
        lambda: model or ScriptedModel(),
        lambda tools: ToolRunner(ToolRegistry(tools), 1, 0),
        retriever or EvidenceRetriever(),
        saver,
        tmp_path / "workflow.jsonl",
        **({"limits": limits} if limits else {}),
    )
    return service
