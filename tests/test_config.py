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
