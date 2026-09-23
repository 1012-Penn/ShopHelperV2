"""HTTP schemas shared by the chat route and orchestration service."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1)
    user_id: str = Field(default="demo-user", min_length=1, max_length=128)
