"""HTTP entry point for MewHelp."""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.db.session import create_tables, make_engine, make_session_factory
from app.schemas import ChatRequest
from app.services.chat import ChatService
from app.tools.registry import ToolRegistry, ToolRunner


ERROR_EVENT = {"event": "error", "data": {"message": "暂时无法处理，请稍后再试。"}}


def create_app(chat_service: ChatService | None = None) -> FastAPI:
    application = FastAPI(title="MewHelp")
    service = chat_service

    def get_chat_service() -> ChatService:
        nonlocal service
        if service is None:
            settings = Settings.from_env()
            engine = make_engine(settings.database_url)
            create_tables(engine)
            session_factory = make_session_factory(engine)
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
            service = ChatService(session_factory, model_factory, runner_factory)
        return service

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

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

    return application


def _encode_sse(event: dict) -> str:
    event_name = event.get("event", "error")
    data = json.dumps(event.get("data", {}), ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"


app = create_app()
