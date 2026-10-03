import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schemas import AfterSaleExtraction, AfterSaleRequest, ChatRequest


def test_settings_maps_deepseek_key_and_knowledge_defaults():
    settings = Settings.from_env(
        {
            "MODEL": "deepseek-chat",
            "API_KEY": "",
            "DEEPSEEK_API_KEY": "test-key",
            "BASE_URL": "https://api.deepseek.com/v1",
            "DATABASE_URL": "sqlite:///test.db",
        }
    )
    assert settings.model == "deepseek-chat"
    assert settings.api_key == "test-key"
    assert settings.embedding_model == "BAAI/bge-m3"


def test_chat_request_rejects_empty_current_message():
    with pytest.raises(ValidationError):
        ChatRequest(conversation_id="c1", message="")


def test_after_sale_request_rejects_blank_text():
    with pytest.raises(ValidationError):
        AfterSaleRequest(text="   ")


def test_after_sale_extraction_allows_missing_fields_as_none():
    result = AfterSaleExtraction()
    assert result.order_id is None
    assert result.request_type is None
    assert result.expected_solution is None
