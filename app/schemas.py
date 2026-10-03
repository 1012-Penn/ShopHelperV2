"""HTTP schemas shared by the chat route and orchestration service."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    category: str | None = Field(default=None, min_length=1, max_length=100)
    conversation_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1)
    user_id: str = Field(default="demo-user", min_length=1, max_length=128)

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value):
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class TicketRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(default="demo-user", min_length=1, max_length=128)
    message_id: int = Field(gt=0)


class AfterSaleRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class AfterSaleExtraction(BaseModel):
    order_id: str | None = None
    request_type: Literal["退款", "退货", "换货", "维修", "物流", "其他"] | None = None
    expected_solution: str | None = None
