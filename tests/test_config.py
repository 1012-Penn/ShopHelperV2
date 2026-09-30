import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app


def test_settings_requires_model_key_and_base_url():
    with pytest.raises(ValueError, match="MODEL"):
        Settings.from_env({"DATABASE_URL": "sqlite:///test.db"})


def test_settings_reads_explicit_values_and_tool_defaults():
    settings = Settings.from_env(
        {
            "MODEL": "demo",
            "API_KEY": "secret-from-env",
            "BASE_URL": "https://model.test/v1",
            "DATABASE_URL": "sqlite:///test.db",
        }
    )
    assert settings.model == "demo"
    assert settings.tool_timeout_seconds == 5
    assert settings.tool_max_retries == 2


def test_health_endpoint_returns_ok():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_seed_can_read_database_url_without_model_credentials():
    assert Settings.database_url_from_env({"DATABASE_URL": "sqlite:///seed.db"}) == "sqlite:///seed.db"


def _chat_env(**updates):
    values = {
        "MODEL": "demo",
        "API_KEY": "chat-secret",
        "BASE_URL": "https://model.test/v1",
        "DATABASE_URL": "sqlite:///test.db",
    }
    values.update(updates)
    return values


def test_settings_reads_knowledge_defaults_and_overrides():
    defaults = Settings.from_env(_chat_env())
    assert defaults.embedding_api_base == "https://api.siliconflow.cn/v1"
    assert defaults.embedding_model == "BAAI/bge-m3"
    assert defaults.embedding_api_key == ""
    assert defaults.milvus_collection == "knowledge"
    assert defaults.knowledge_max_chars == 1200
    assert defaults.knowledge_overlap_chars == 200
    assert defaults.faq_top_k == 5
    assert defaults.faq_min_similarity == 0.40
    assert defaults.knowledge_batch_size == 32

    configured = Settings.from_env(
        _chat_env(
            EMBEDDING_API_BASE="https://embed.test/v1",
            EMBEDDING_MODEL="custom-bge-m3",
            EMBEDDING_API_KEY="embedding-secret",
            MILVUS_URI="http://milvus.test:19530",
            MILVUS_COLLECTION="support_kb",
            KNOWLEDGE_MAX_CHARS="900",
            KNOWLEDGE_OVERLAP_CHARS="120",
            FAQ_TOP_K="7",
            FAQ_MIN_SIMILARITY="0.61",
            KNOWLEDGE_BATCH_SIZE="16",
        )
    )
    assert configured.embedding_api_base == "https://embed.test/v1"
    assert configured.embedding_model == "custom-bge-m3"
    assert configured.embedding_api_key == "embedding-secret"
    assert configured.milvus_uri == "http://milvus.test:19530"
    assert configured.milvus_collection == "support_kb"
    assert configured.knowledge_max_chars == 900
    assert configured.knowledge_overlap_chars == 120
    assert configured.faq_top_k == 7
    assert configured.faq_min_similarity == 0.61
    assert configured.knowledge_batch_size == 16


@pytest.mark.parametrize(
    "updates",
    [
        {"KNOWLEDGE_MAX_CHARS": "200", "KNOWLEDGE_OVERLAP_CHARS": "200"},
        {"KNOWLEDGE_MAX_CHARS": "100", "KNOWLEDGE_OVERLAP_CHARS": "200"},
        {"FAQ_TOP_K": "0"},
        {"FAQ_TOP_K": "21"},
        {"FAQ_MIN_SIMILARITY": "-0.01"},
        {"FAQ_MIN_SIMILARITY": "1.01"},
        {"KNOWLEDGE_BATCH_SIZE": "0"},
    ],
)
def test_settings_rejects_invalid_knowledge_bounds(updates):
    with pytest.raises(ValueError):
        Settings.from_env(_chat_env(**updates))


def test_knowledge_credentials_are_required_only_when_requested():
    settings = Settings.from_env(_chat_env())
    assert settings.model == "demo"
    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        settings.require_knowledge()

    with pytest.raises(ValueError, match="MILVUS_URI"):
        Settings.from_env(
            _chat_env(EMBEDDING_API_KEY="embedding-secret")
        ).require_knowledge()
