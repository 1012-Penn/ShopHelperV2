from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model: str = Field(validation_alias="MODEL")
    api_key: str = Field(
        validation_alias=AliasChoices("API_KEY", "DEEPSEEK_API_KEY")
    )
    base_url: str = Field(validation_alias="BASE_URL")
    max_history_tokens: int = Field(default=1024, validation_alias="MAX_HISTORY_TOKENS")

    model_config = SettingsConfigDict(
        env_file=(Path("/root/.env"), Path(".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )
