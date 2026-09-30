"""Factories for short-lived knowledge CLI processes."""

from app.config import Settings
from app.db.session import create_tables, make_engine, make_session_factory
from app.services.knowledge.conversations import ConversationKnowledgeExtractor, LangChainConversationModel
from app.services.knowledge.embeddings import EmbeddingClient
from app.services.knowledge.indexer import KnowledgeIndexer
from app.services.knowledge.repository import KnowledgeRepository
from app.services.knowledge.vector_store import MilvusKnowledgeStore


def build_indexer(settings: Settings | None = None) -> KnowledgeIndexer:
    settings = settings or Settings.from_env(require_chat=False)
    settings.require_knowledge()
    engine = make_engine(settings.database_url)
    embeddings = None
    vector_store = None
    try:
        create_tables(engine)
        repository = KnowledgeRepository(make_session_factory(engine))
        embeddings = EmbeddingClient(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_api_base,
            model=settings.embedding_model,
        )
        vector_store = MilvusKnowledgeStore(
            uri=settings.milvus_uri,
            collection_name=settings.milvus_collection,
        )
        vector_store.ensure_collection()
        indexer = KnowledgeIndexer(repository, embeddings, vector_store)
        indexer.settings = settings
        return indexer
    except Exception:
        for service in (embeddings, vector_store):
            close = getattr(service, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        engine.dispose()
        raise


def build_extractor() -> tuple[ConversationKnowledgeExtractor, KnowledgeIndexer]:
    settings = Settings.from_env(allow_chat_key_fallback=False)
    indexer = build_indexer(settings)
    try:
        model = LangChainConversationModel.from_settings(settings)
        extractor = ConversationKnowledgeExtractor(model, indexer.repository)
        return extractor, indexer
    except Exception:
        indexer.close()
        raise
