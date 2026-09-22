from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.prompts import build_chat_prompt
from app.schemas import ChatRequest, Message


def _to_langchain_message(message: Message) -> BaseMessage:
    if message.role == "user":
        return HumanMessage(content=message.content)
    return AIMessage(content=message.content)


def trim_history(history: list[Message], max_tokens: int) -> list[BaseMessage]:
    if not history:
        return []

    messages = [_to_langchain_message(message) for message in history]
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        end_on=("human", "ai"),
        include_system=False,
        allow_partial=False,
    )


def build_chat_model(settings: Settings) -> ChatOpenAI:
    if not settings.model or not settings.api_key or not settings.base_url:
        raise ValueError("MODEL, API_KEY, and BASE_URL must be configured")
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        streaming=True,
    )


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    content = getattr(chunk, "content", "")
    return content if isinstance(content, str) else str(content)


class ChatService:
    def __init__(self, model: Any, settings: Settings):
        self.model = model
        self.settings = settings
        self.prompt = build_chat_prompt()

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        history = trim_history(
            request.history,
            max_tokens=self.settings.max_history_tokens,
        )
        messages = self.prompt.invoke(
            {"history": history, "message": request.message}
        ).to_messages()
        async for chunk in self.model.astream(messages):
            text = _chunk_text(chunk)
            if text:
                yield text
