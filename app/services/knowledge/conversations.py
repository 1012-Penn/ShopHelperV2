"""Scheduled extraction of reusable ecommerce support knowledge from chat history."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.services.knowledge.privacy import redact_customer_data
from app.services.knowledge.repository import ConversationTurn, KnowledgeRepository


class ExtractedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_turn_index: int = Field(ge=0)
    category: str = Field(min_length=1, max_length=256)
    questions: list[str] = Field(min_length=1, max_length=8)
    answer: str = Field(min_length=1)


class ExtractionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[ExtractedCandidate]


@dataclass(frozen=True)
class ExtractionSummary:
    messages_read: int
    staged_pairs: int
    deduped: int
    inserted: int
    skipped_tool_messages: int
    last_message_id: int


class LangChainConversationModel:
    """LangChain structured-output adapter for the configured support chat model."""

    def __init__(self, model, api_key: str, base_url: str):
        from langchain_openai import ChatOpenAI

        self._structured_model = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        ).with_structured_output(ExtractionBatch, method="function_calling")

    @classmethod
    def from_settings(cls, settings: Settings) -> "LangChainConversationModel":
        return cls(settings.model, settings.api_key, settings.base_url)

    def extract_pairs(self, turns: list[ConversationTurn]) -> list[ExtractedCandidate]:
        payload = [
            {
                "source_turn_index": index,
                "customer_question": turn.user_text,
                "support_answer": turn.assistant_text,
            }
            for index, turn in enumerate(turns)
        ]
        result = self._structured_model.invoke(
            [
                (
                    "system",
                    "你为电商客服知识库筛选可复用的问答。只提取已经明确解决、可推广到其他顾客的问题。"
                    "不要保留顾客姓名、联系方式、订单号、个案信息；不要编造店铺政策、费率或时限。"
                    "回答不确定、只适用于单个订单、没有实际答案或含有工具内部内容的对话应跳过。"
                    "保留原始用户问法作为 questions，并只引用已确认的客服答复。",
                ),
                ("human", json.dumps(payload, ensure_ascii=False)),
            ]
        )
        if isinstance(result, ExtractionBatch):
            return result.candidates
        if isinstance(result, dict):
            return ExtractionBatch.model_validate(result).candidates
        raise ValueError("structured extraction response is missing or invalid")


class ConversationKnowledgeExtractor:
    def __init__(self, model, repository: KnowledgeRepository):
        self.model = model
        self.repository = repository

    def run(self, batch_size: int) -> ExtractionSummary:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        recovered_deduped, recovered_inserted = self.repository.promote_staged()
        cursor = self.repository.extraction_cursor()
        turns, next_cursor = self.repository.load_turns_after(cursor, batch_size)
        if next_cursor <= cursor:
            return ExtractionSummary(0, 0, recovered_deduped, recovered_inserted, 0, cursor)

        skipped_tool_messages = self.repository.count_skipped_tool_messages(cursor, next_cursor)
        if turns:
            sanitized_turns = [
                replace(
                    turn,
                    user_text=redact_customer_data(turn.user_text),
                    assistant_text=redact_customer_data(turn.assistant_text),
                )
                for turn in turns
            ]
            candidates = self.model.extract_pairs(sanitized_turns)
        else:
            sanitized_turns = []
            candidates = []

        mapped: list[tuple[ConversationTurn, ExtractedCandidate]] = []
        for raw_candidate in candidates:
            candidate = (
                raw_candidate
                if isinstance(raw_candidate, ExtractedCandidate)
                else ExtractedCandidate.model_validate(raw_candidate)
            )
            if candidate.source_turn_index >= len(turns):
                raise ValueError("extraction response references a turn outside the current batch")
            category = redact_customer_data(candidate.category)
            questions = [redact_customer_data(question) for question in candidate.questions]
            answer = redact_customer_data(candidate.answer)
            if not category.strip() or not questions or any(not question.strip() for question in questions) or not answer.strip():
                raise ValueError("extraction response contains an empty knowledge field")
            safe_candidate = candidate.model_copy(
                update={"category": category, "questions": questions, "answer": answer}
            )
            mapped.append((turns[candidate.source_turn_index], safe_candidate))

        run_id = uuid4().hex
        self.repository.stage_batch_and_advance(mapped, next_cursor, run_id)
        deduped, inserted = self.repository.promote_staged()
        return ExtractionSummary(
            messages_read=self.repository.count_messages_between(cursor, next_cursor),
            staged_pairs=len(mapped),
            deduped=recovered_deduped + deduped,
            inserted=recovered_inserted + inserted,
            skipped_tool_messages=skipped_tool_messages,
            last_message_id=next_cursor,
        )
