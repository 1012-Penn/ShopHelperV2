from typing import Literal

from pydantic import BaseModel, Field, field_validator


MessageRole = Literal["user", "assistant"]
RequestType = Literal["退款", "退货", "换货", "维修", "物流", "其他"]


class Message(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content must not be blank")
        return value


class ChatRequest(BaseModel):
    conversation_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    history: list[Message] = Field(default_factory=list)

    @field_validator("conversation_id", "message")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


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
    request_type: RequestType | None = None
    expected_solution: str | None = None
