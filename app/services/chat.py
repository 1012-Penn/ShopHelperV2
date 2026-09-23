"""Persisted, single-tool-call chat orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Conversation, Message
from app.prompts import SYSTEM_PROMPT
from app.schemas import ChatRequest
from app.tools.business import build_tools
from app.tools.registry import ToolInputError, ToolResult, ToolRunner, UnknownToolError

ToolRunnerFactory = Callable[[list], ToolRunner]
ModelFactory = Callable[[], object]


class ChatService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        model_factory: ModelFactory,
        tool_runner_factory: ToolRunnerFactory,
    ) -> None:
        self.session_factory = session_factory
        self.model_factory = model_factory
        self.tool_runner_factory = tool_runner_factory

    def stream_events(self, request: ChatRequest) -> Iterator[dict]:
        try:
            history = self._persist_user_and_load_history(request)
            tools = build_tools(self.session_factory, request.conversation_id)
            runner = self.tool_runner_factory(tools)
            model = self.model_factory()
            messages = [SystemMessage(content=SYSTEM_PROMPT), *history, HumanMessage(content=request.message)]
            chunks = list(model.bind_tools(runner.registry.tools).stream(messages))
            response = self._combine_chunks(chunks)
            tool_calls = response.tool_calls

            if len(tool_calls) > 1:
                yield self._error_event()
                return

            if not tool_calls:
                answer_parts = [self._chunk_text(chunk) for chunk in chunks]
                answer_parts = [part for part in answer_parts if part]
                for part in answer_parts:
                    yield {"event": "token", "data": {"content": part}}
                answer = "".join(answer_parts)
                self._persist_assistant(request.conversation_id, answer)
                yield {"event": "done", "data": {"conversation_id": request.conversation_id}}
                return

            call = tool_calls[0]
            tool_name = call["name"]
            tool_call_id = call.get("id") or str(uuid4())
            args = call.get("args") or {}
            assistant_message = AIMessage(content=response.content or "", tool_calls=[call])
            self._persist_tool_request(request.conversation_id, assistant_message, tool_call_id)
            yield {"event": "tool_status", "data": {"tool_name": tool_name, "status": "running"}}

            try:
                result = runner.run(tool_name, args, tool_call_id)
            except (ToolInputError, UnknownToolError):
                result = ToolResult(tool_name, tool_call_id, "工具请求无效，无法执行。", True)
            self._persist_tool_result(request.conversation_id, result)

            final_messages = [
                *messages,
                assistant_message,
                ToolMessage(content=result.content, tool_call_id=tool_call_id),
            ]
            answer_parts: list[str] = []
            for chunk in model.stream(final_messages):
                part = self._chunk_text(chunk)
                if part:
                    answer_parts.append(part)
                    yield {"event": "token", "data": {"content": part}}
            answer = "".join(answer_parts)
            self._persist_assistant(request.conversation_id, answer)
            yield {"event": "done", "data": {"conversation_id": request.conversation_id}}
        except Exception:
            yield self._error_event()

    def _persist_user_and_load_history(self, request: ChatRequest) -> list:
        with self.session_factory.begin() as session:
            conversation = session.get(Conversation, request.conversation_id)
            if conversation is None:
                conversation = Conversation(
                    conversation_id=request.conversation_id,
                    user_id=request.user_id,
                    status="open",
                )
                session.add(conversation)
                session.flush()
                previous = []
            else:
                if conversation.user_id != request.user_id:
                    raise ValueError("Conversation does not belong to this user")
                previous = list(
                    session.scalars(
                        select(Message)
                        .where(Message.conversation_id == request.conversation_id)
                        .order_by(Message.id)
                    )
                )
            session.add(
                Message(
                    conversation_id=request.conversation_id,
                    role="user",
                    content=request.message,
                )
            )
        return [self._to_langchain_message(message) for message in previous]

    @staticmethod
    def _to_langchain_message(message: Message):
        if message.role == "user":
            return HumanMessage(content=message.content)
        if message.role == "tool":
            return ToolMessage(content=message.content, tool_call_id=message.tool_call_id or "missing-tool-id")
        if message.tool_calls:
            return AIMessage(content=message.content, tool_calls=message.tool_calls)
        return AIMessage(content=message.content)

    @staticmethod
    def _combine_chunks(chunks: list[AIMessageChunk]) -> AIMessageChunk:
        if not chunks:
            return AIMessageChunk(content="")
        response = chunks[0]
        for chunk in chunks[1:]:
            response = response + chunk
        return response

    @staticmethod
    def _chunk_text(chunk) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )

    def _persist_tool_request(self, conversation_id: str, message: AIMessage, tool_call_id: str) -> None:
        with self.session_factory.begin() as session:
            session.add(
                Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=str(message.content or ""),
                    tool_calls=message.tool_calls,
                    tool_call_id=tool_call_id,
                )
            )

    def _persist_tool_result(self, conversation_id: str, result: ToolResult) -> None:
        content = result.content
        if result.is_error:
            content = json.dumps({"error": True, "message": result.content}, ensure_ascii=False)
        with self.session_factory.begin() as session:
            session.add(
                Message(
                    conversation_id=conversation_id,
                    role="tool",
                    content=content,
                    tool_call_id=result.tool_call_id,
                )
            )

    def _persist_assistant(self, conversation_id: str, answer: str) -> None:
        with self.session_factory.begin() as session:
            session.add(Message(conversation_id=conversation_id, role="assistant", content=answer))

    @staticmethod
    def _error_event() -> dict:
        return {"event": "error", "data": {"message": "暂时无法处理，请稍后再试。"}}
