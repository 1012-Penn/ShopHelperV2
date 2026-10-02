"""Shared production factories for chat, rebuild and live evaluation."""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from app.config import Settings
from app.services.knowledge.embeddings import EmbeddingClient
from app.services.knowledge.hybrid_store import HybridStore
from app.services.quality.query import QueryNormalizer
from app.services.quality.rerank import Reranker
from app.services.quality.retrieval import QualityRetriever
from app.services.quality.generation import KnowledgeAnswerService, StructuredGenerator
from app.services.quality.generation_diagnostics import JsonlGenerationFailureSink
from app.services.quality.ledger import QualityLedger


def config():
    return {**dotenv_values('.env'),**os.environ}


def build_answer_service(session_factory, settings=None):
    settings=settings or Settings.from_env()
    values=config()
    embeddings=EmbeddingClient(api_key=settings.embedding_api_key,base_url=settings.embedding_api_base,model=settings.embedding_model)
    store=HybridStore(uri=settings.milvus_uri or 'http://localhost:19530',collection_name=values.get('HYBRID_COLLECTION','knowledge_ch04'))
    store.ensure_collection()
    reranker=Reranker(values.get('RERANK_API_KEY',''),values.get('RERANK_API_BASE','https://api.siliconflow.cn/v1'))
    model=ChatOpenAI(model=settings.model,api_key=settings.api_key,base_url=settings.base_url,temperature=0,timeout=90,max_retries=1)
    normalizer=QueryNormalizer.from_model(model)
    retriever=QualityRetriever(session_factory,embeddings,store,reranker,normalizer)
    project_root=Path(__file__).resolve().parents[3]
    diagnostic_path=Path(values.get('QUALITY_GENERATION_DIAGNOSTICS_PATH') or '.runtime/quality/generation-failures.jsonl')
    if not diagnostic_path.is_absolute():
        diagnostic_path=project_root/diagnostic_path
    return KnowledgeAnswerService(
        retriever,
        StructuredGenerator(model,model_identity=settings.model),
        QualityLedger(session_factory),
        min_score=float(values.get('RERANK_MIN_SCORE','0.05')),
        diagnostic_sink=JsonlGenerationFailureSink(diagnostic_path),
        model_identity=settings.model,
    )
