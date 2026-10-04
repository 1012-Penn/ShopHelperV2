"""HTTP schemas shared by the chat route and orchestration service."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ChatRequest(BaseModel):
    category: str | None = Field(default=None, min_length=1, max_length=100)
    conversation_id: str = Field(min_length=1, max_length=128)
    message: str = Field(default="")
    user_id: str = Field(default="demo-user", min_length=1, max_length=128)
    selected_order_id: str | None = Field(default=None, min_length=1, max_length=128)
    selection_message_id: int | None = Field(default=None, gt=0)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_chat_or_order_selection(self):
        resume_fields = (
            self.selected_order_id,
            self.selection_message_id,
            self.request_id,
        )
        if any(value is not None for value in resume_fields) and not all(
            value is not None for value in resume_fields
        ):
            raise ValueError("order selection fields must be provided together")
        if not all(value is not None for value in resume_fields) and not self.message.strip():
            raise ValueError("message must not be blank")
        return self


class TicketRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(default="demo-user", min_length=1, max_length=128)
    message_id: int = Field(gt=0)


class RefundApplicationRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(default="demo-user", min_length=1, max_length=128)
    message_id: int = Field(gt=0)
    request_id: str = Field(min_length=1, max_length=128)
    order_id: str = Field(min_length=1, max_length=128)
    request_type: Literal["退款", "退货"]
    reason: Literal["商品质量问题", "错发/漏发", "不想要/不合适", "其他"]


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
