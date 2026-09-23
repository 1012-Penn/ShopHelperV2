from collections.abc import AsyncIterable
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.schemas import AfterSaleExtraction, AfterSaleRequest, ChatRequest
from app.services.after_sale import AfterSaleService
from app.services.chat import ChatService, build_chat_model

app = FastAPI(title="MewHelp")


def get_chat_service() -> ChatService:
    settings = Settings()
    return ChatService(model=build_chat_model(settings), settings=settings)


def get_after_sale_service() -> AfterSaleService:
    settings = Settings()
    return AfterSaleService(model=build_chat_model(settings))


@app.post("/api/v1/chat/stream", response_class=EventSourceResponse)
async def chat_stream(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),  # noqa: B008
) -> AsyncIterable[ServerSentEvent]:
    try:
        async for chunk in service.stream(request):
            yield ServerSentEvent(
                event="token",
                data={"content": chunk},
            )
    except Exception:  # noqa: BLE001 - upstream failures must become SSE errors
        yield ServerSentEvent(
            event="error",
            data={"message": "上游模型调用失败"},
        )
        return

    yield ServerSentEvent(
        event="done",
        data={"conversation_id": request.conversation_id},
    )


@app.post("/api/v1/after-sale/extract", response_model=AfterSaleExtraction)
def extract_after_sale(
    request: AfterSaleRequest,
    service: AfterSaleService = Depends(get_after_sale_service),  # noqa: B008
) -> AfterSaleExtraction:
    return service.extract(request.text)


frontend_root = Path(__file__).resolve().parent.parent / "frontend"
frontend_dist = frontend_root / "dist"
app.mount(
    "/",
    StaticFiles(directory=frontend_dist if frontend_dist.exists() else frontend_root, html=True),
    name="frontend",
)
