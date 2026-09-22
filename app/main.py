from collections.abc import AsyncIterable

from fastapi import Depends, FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.config import Settings
from app.schemas import ChatRequest
from app.services.chat import ChatService, build_chat_model

app = FastAPI(title="MewHelp")


def get_chat_service() -> ChatService:
    settings = Settings()
    return ChatService(model=build_chat_model(settings), settings=settings)


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
