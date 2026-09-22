import pytest


def test_trim_history_keeps_latest_complete_turn():
    from app.schemas import Message
    from app.services.chat import trim_history

    history = [
        Message(role="user", content="旧问题"),
        Message(role="assistant", content="旧回答"),
        Message(role="user", content="最近问题"),
        Message(role="assistant", content="最近回答"),
    ]
    kept = trim_history(history, max_tokens=12)
    assert kept[-1].content == "最近回答"
    assert kept[0].type == "human"


def test_trim_history_handles_empty_and_incomplete_history():
    from app.schemas import Message
    from app.services.chat import trim_history

    assert trim_history([], max_tokens=12) == []
    kept = trim_history(
        [Message(role="assistant", content="孤立回答")],
        max_tokens=12,
    )
    assert kept == []


def test_prompt_contains_customer_service_constraints():
    from app.prompts import build_chat_prompt

    system_text = build_chat_prompt().messages[0].prompt.template
    assert "电商客服" in system_text
    assert "不得编造" in system_text


@pytest.mark.asyncio
async def test_chat_service_streams_fake_model_chunks_and_preserves_context():
    from app.config import Settings
    from app.schemas import ChatRequest
    from app.services.chat import ChatService

    class FakeModel:
        def __init__(self):
            self.received = None

        async def astream(self, messages):
            self.received = messages
            yield "您好"
            yield "，请提供订单号。"

    model = FakeModel()
    settings = Settings(
        _env_file=None,
        MODEL="test-model",
        API_KEY="test-key",
        BASE_URL="https://example.test/v1",
        MAX_HISTORY_TOKENS=128,
    )
    service = ChatService(model=model, settings=settings)
    request = ChatRequest(
        conversation_id="c1",
        message="我的订单还没收到",
        history=[{"role": "user", "content": "我想查物流"}],
    )

    chunks = [chunk async for chunk in service.stream(request)]

    assert chunks == ["您好", "，请提供订单号。"]
    assert model.received[0].type == "system"
    assert model.received[-1].content == "我的订单还没收到"
