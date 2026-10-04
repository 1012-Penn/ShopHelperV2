"""HTTP entry point for MewHelp."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.schemas import (
    AfterSaleExtraction,
    AfterSaleRequest,
    ChatRequest,
    RefundApplicationRequest,
    TicketRequest,
)
from app.services.after_sale import AfterSaleService
from app.services.chat import ChatService
from app.services.workflow.runtime import build_workflow_service
from app.services.workflow.refunds import (
    DemoOrderError,
    DemoRefundApplications,
    RefundConflict,
    RefundOfferError,
)

ERROR_EVENT = {"event": "error", "data": {"message": "暂时无法处理，请稍后再试。"}}


def get_after_sale_service() -> AfterSaleService:
    settings = Settings.from_env()
    model = ChatOpenAI(
        model=settings.model, api_key=settings.api_key, base_url=settings.base_url
    )
    return AfterSaleService(model)


def create_app(chat_service: ChatService | None = None) -> FastAPI:
    service = chat_service
    service_lock = Lock()

    @asynccontextmanager
    async def lifespan(application):
        try:
            yield
        finally:
            if service is not None and hasattr(service, "close"):
                service.close()

    application = FastAPI(title="MewHelp", lifespan=lifespan)

    def get_chat_service() -> ChatService:
        nonlocal service
        with service_lock:
            if service is None:
                service = build_workflow_service()
        return service

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/", include_in_schema=False)
    def chat_page() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @application.post("/api/v1/tickets")
    def create_ticket_from_button(request: TicketRequest):
        try:
            chat = get_chat_service()
            if not hasattr(chat, "ticket_actions"):
                raise HTTPException(status_code=503, detail="工单服务暂不可用")
            return chat.ticket_actions.submit(
                request.conversation_id, request.user_id, request.message_id
            )
        except ValueError as error:
            raise HTTPException(status_code=403, detail="会话或工单建议无效") from error
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail="工单结果暂未确认，请勿重复提交；请联系人工客服核查。",
            ) from error

    @application.post("/api/v1/refund-applications")
    def create_demo_refund_application(request: RefundApplicationRequest):
        try:
            chat = get_chat_service()
            if not hasattr(chat, "session_factory"):
                raise HTTPException(status_code=503, detail="演示退款服务暂不可用")
            return DemoRefundApplications(chat.session_factory).submit(request)
        except RefundConflict as error:
            raise HTTPException(status_code=409, detail="请求标识已用于其他申请内容") from error
        except RefundOfferError as error:
            raise HTTPException(status_code=403, detail="会话或退款表单无效") from error
        except DemoOrderError as error:
            raise HTTPException(status_code=422, detail="演示订单无效") from error
        except HTTPException:
            raise
        except Exception as error:
            logging.getLogger(__name__).error(
                "Demo refund submission failed (%s)", type(error).__name__
            )
            raise HTTPException(status_code=503, detail="演示申请暂未确认，请勿重复提交") from error

    @application.get("/api/v1/knowledge/source", response_class=HTMLResponse)
    def knowledge_source(source: str):
        from sqlalchemy import select

        from app.db.models import KnowledgeChunk
        from app.services.quality.sources import SourceDocuments

        chat = get_chat_service()
        with chat.session_factory() as session:
            keys = list(
                session.scalars(
                    select(KnowledgeChunk.source_key).where(
                        KnowledgeChunk.is_active.is_(True)
                    )
                )
            )
        registered = {
            key[4:].rsplit(":", 1)[0] for key in keys if key.startswith("doc:")
        }
        root = Path(__file__).resolve().parent.parent / "knowledge_docs"
        try:
            return HTMLResponse(SourceDocuments(root, registered).render(source))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="来源原文不可用") from None

    @application.post("/api/v1/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        try:
            chat = get_chat_service()
        except Exception as error:
            raise HTTPException(
                status_code=503, detail=ERROR_EVENT["data"]["message"]
            ) from error

        def event_stream() -> Iterator[str]:
            try:
                for event in chat.stream_events(request):
                    yield _encode_sse(event)
            except Exception as error:  # noqa: BLE001 - convert stream failures to SSE error
                logging.getLogger(__name__).error(
                    "SSE stream failed (%s)", type(error).__name__
                )
                yield _encode_sse(ERROR_EVENT)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.post("/api/v1/after-sale/extract", response_model=AfterSaleExtraction)
    def extract_after_sale(
        request: AfterSaleRequest,
        after_sale: AfterSaleService = Depends(get_after_sale_service),  # noqa: B008 - FastAPI dependency
    ) -> AfterSaleExtraction:
        return after_sale.extract(request.text)

    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        application.mount(
            "/assets",
            StaticFiles(directory=frontend_dist / "assets"),
            name="frontend-assets",
        )

    return application


def _encode_sse(event: dict) -> str:
    event_name = event.get("event", "error")
    data = json.dumps(event.get("data", {}), ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"


app = create_app()
