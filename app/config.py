from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model: str = Field(validation_alias="MODEL")
    api_key: str = Field(default="", validation_alias="API_KEY")
    deepseek_api_key: str | None = Field(
        default=None,
        validation_alias="DEEPSEEK_API_KEY",
    )
    base_url: str = Field(validation_alias="BASE_URL")
    max_history_tokens: int = Field(default=1024, validation_alias="MAX_HISTORY_TOKENS")

    model_config = SettingsConfigDict(
        env_file=(Path("/root/.env"), Path(".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def prefer_deepseek_key_when_api_key_is_blank(self):
        if not self.api_key and self.deepseek_api_key:
            self.api_key = self.deepseek_api_key
        return self
