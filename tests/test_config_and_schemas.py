import pytest
from pydantic import ValidationError


def test_settings_maps_deepseek_key_and_defaults(monkeypatch):
    monkeypatch.setenv("MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://api.deepseek.com/v1")

    from app.config import Settings

    settings = Settings(_env_file=None)
    assert settings.model == "deepseek-chat"
    assert settings.api_key == "test-key"
    assert settings.max_history_tokens > 0


def test_chat_request_rejects_empty_current_message():
    from app.schemas import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(conversation_id="c1", message="", history=[])


def test_chat_request_accepts_valid_history():
    from app.schemas import ChatRequest

    request = ChatRequest(
        conversation_id="c1",
        message="请继续处理",
        history=[
            {"role": "user", "content": "我想退货"},
            {"role": "assistant", "content": "请提供订单号"},
        ],
    )
    assert request.history[0].role == "user"


def test_message_rejects_unknown_role():
    from app.schemas import Message

    with pytest.raises(ValidationError):
        Message(role="system", content="不允许")


def test_after_sale_extraction_allows_missing_fields_as_none():
    from app.schemas import AfterSaleExtraction

    result = AfterSaleExtraction()
    assert result.order_id is None
    assert result.request_type is None
    assert result.expected_solution is None
