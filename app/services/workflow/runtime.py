"""Single-process production factory; knowledge dependencies initialize on demand."""

import sqlite3
from pathlib import Path
from threading import Lock

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import Settings
from app.db.session import create_tables, make_engine, make_session_factory
from app.services.context.budget import ContextBudget
from app.services.quality.runtime import build_retriever, config
from app.tools.registry import ToolRegistry, ToolRunner

from .graph import WorkflowService
from .policy import Limits
from .retrieval import EvidenceAdapter


class LazyRetriever:
    def __init__(self, factory):
        self.factory, self.value, self.lock = factory, None, Lock()

    def retrieve(self, question, category=None):
        with self.lock:
            if self.value is None:
                self.value = self.factory()
        return self.value.retrieve(question, category)

    def close(self):
        if self.value is not None:
            store = getattr(self.value.retriever, "store", None)
            if store is None:
                store = getattr(
                    getattr(self.value.retriever, "searcher", None),
                    "vector_store",
                    None,
                )
            if hasattr(store, "close"):
                store.close()


def build_workflow_service(settings=None, values=None):
    settings = settings or Settings.from_env()
    values = config() if values is None else values
    budget = ContextBudget.from_env(values)
    limits = Limits(
        **{
            key: int(values.get(env, default))
            for key, env, default in (
                ("max_decisions", "MAX_AGENT_STEPS", budget.steps),
                ("max_tool_calls", "AGENT_MAX_TOOL_CALLS", 6),
                ("max_output_tokens", "MAX_OUTPUT_TOKENS", budget.output),
                ("max_tokens", "AGENT_MAX_TOKENS", 1000000),
            )
        }
    )
    quality_enabled = str(values.get("QUALITY_ENABLED", "true")).lower() == "true"
    threshold = (
        float(values.get("RERANK_MIN_SCORE", "0.05"))
        if quality_enabled
        else settings.faq_min_similarity
    )
    root = Path(__file__).resolve().parents[3]

    def path_for(key, default):
        path = Path(values.get(key) or default)
        return path if path.is_absolute() else root / path

    checkpoint_path = path_for(
        "WORKFLOW_CHECKPOINT_PATH", ".runtime/ch05/checkpoints.sqlite"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.database_url)
    conn = None
    try:
        create_tables(engine)
        session_factory = make_session_factory(engine)

        def load_retriever():
            settings.require_knowledge()
            if quality_enabled:
                return EvidenceAdapter(build_retriever(session_factory, settings))
            from app.services.knowledge.embeddings import EmbeddingClient
            from app.services.knowledge.repository import KnowledgeRepository
            from app.services.knowledge.retriever import (
                DenseSearcher,
                KnowledgeRetriever,
            )
            from app.services.knowledge.vector_store import MilvusKnowledgeStore

            store = MilvusKnowledgeStore(
                uri=settings.milvus_uri, collection_name=settings.milvus_collection
            )
            store.ensure_collection()
            embeddings = EmbeddingClient(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_api_base,
                model=settings.embedding_model,
            )
            retriever = KnowledgeRetriever(
                DenseSearcher(embeddings, store),
                KnowledgeRepository(session_factory),
                settings.faq_top_k,
                settings.faq_min_similarity,
            )
            return EvidenceAdapter(retriever, "dense")

        retriever = LazyRetriever(load_retriever)
        conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
        model_factory = lambda: ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=0,
            timeout=45,
            max_retries=0,
            stream_usage=True,
        )
        runner_factory = lambda tools: ToolRunner(
            ToolRegistry(tools),
            settings.tool_timeout_seconds,
            settings.tool_max_retries,
        )
        service = WorkflowService(
            session_factory,
            model_factory,
            runner_factory,
            retriever,
            SqliteSaver(conn),
            path_for("WORKFLOW_LOG_PATH", ".runtime/ch05/workflow.jsonl"),
            limits=limits,
            min_score=threshold,
            strategy="hybrid_rerank" if quality_enabled else "dense",
            context_budget=budget,
            context_log_path=path_for("CONTEXT_LOG_PATH", "log/app.log"),
        )
        service._closers.extend([retriever.close, engine.dispose])
        return service
    except Exception:
        if conn:
            conn.close()
        engine.dispose()
        raise
