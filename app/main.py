"""HTTP entry point for MewHelp."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.db.session import create_tables, make_engine, make_session_factory
from app.schemas import AfterSaleExtraction, AfterSaleRequest, ChatRequest
from app.services.after_sale import AfterSaleService
from app.services.chat import ChatService
from app.services.knowledge.embeddings import EmbeddingClient
from app.services.knowledge.repository import KnowledgeRepository
from app.services.knowledge.retriever import DenseSearcher, KnowledgeRetriever
from app.services.knowledge.vector_store import MilvusKnowledgeStore
from app.tools.registry import ToolRegistry, ToolRunner


ERROR_EVENT = {"event": "error", "data": {"message": "暂时无法处理，请稍后再试。"}}


def get_after_sale_service() -> AfterSaleService:
    settings = Settings.from_env()
    model = ChatOpenAI(model=settings.model, api_key=settings.api_key, base_url=settings.base_url)
    return AfterSaleService(model)


def create_app(chat_service: ChatService | None = None) -> FastAPI:
    application = FastAPI(title="MewHelp")
    service = chat_service

    def get_chat_service() -> ChatService:
        nonlocal service
        if service is None:
            settings = Settings.from_env()
            settings.require_knowledge()
            engine = make_engine(settings.database_url)
            create_tables(engine)
            session_factory = make_session_factory(engine)
            embedding_client = EmbeddingClient(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_api_base,
                model=settings.embedding_model,
            )
            vector_store = MilvusKnowledgeStore(
                uri=settings.milvus_uri,
                collection_name=settings.milvus_collection,
            )
            vector_store.ensure_collection()
            faq_retriever = KnowledgeRetriever(
                DenseSearcher(embedding_client, vector_store),
                KnowledgeRepository(session_factory),
                top_k=settings.faq_top_k,
                min_similarity=settings.faq_min_similarity,
            )
            model_factory = lambda: ChatOpenAI(
                model=settings.model,
                api_key=settings.api_key,
                base_url=settings.base_url,
            )
            runner_factory = lambda tools: ToolRunner(
                ToolRegistry(tools),
                timeout_seconds=settings.tool_timeout_seconds,
                max_retries=settings.tool_max_retries,
            )
            service = ChatService(session_factory, model_factory, runner_factory, faq_retriever=faq_retriever)
        return service

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/", include_in_schema=False)
    def chat_page() -> FileResponse:
        frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
        page = frontend_dist / "index.html"
        if not page.exists():
            page = Path(__file__).parent / "static" / "index.html"
        return FileResponse(page)

    @application.post("/api/v1/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        try:
            chat = get_chat_service()
        except Exception as error:
            raise HTTPException(status_code=503, detail=ERROR_EVENT["data"]["message"]) from error

        def event_stream() -> Iterator[str]:
            try:
                for event in chat.stream_events(request):
                    yield _encode_sse(event)
            except Exception:
                yield _encode_sse(ERROR_EVENT)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.post("/api/v1/after-sale/extract", response_model=AfterSaleExtraction)
    def extract_after_sale(
        request: AfterSaleRequest,
        after_sale: AfterSaleService = Depends(get_after_sale_service),
    ) -> AfterSaleExtraction:
        return after_sale.extract(request.text)

    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        application.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="frontend-assets")

    return application


def _encode_sse(event: dict) -> str:
    event_name = event.get("event", "error")
    data = json.dumps(event.get("data", {}), ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"


app = create_app()
