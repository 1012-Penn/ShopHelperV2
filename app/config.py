"""Runtime configuration for the customer support demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    model: str
    api_key: str
    base_url: str
    database_url: str
    tool_timeout_seconds: float = 5
    tool_max_retries: int = 2

    @staticmethod
    def database_url_from_env(environ: Mapping[str, str] | None = None) -> str:
        if environ is None:
            values = {key: value for key, value in dotenv_values(Path.cwd() / ".env").items() if value is not None}
            values.update(os.environ)
        else:
            values = environ
        database_url = values.get("DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        return database_url

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        if environ is None:
            values = {key: value for key, value in dotenv_values(Path.cwd() / ".env").items() if value is not None}
            values.update(os.environ)
            if not values.get("API_KEY") and not values.get("DEEPSEEK_API_KEY"):
                fallback = dotenv_values("/root/.env").get("DEEPSEEK_API_KEY")
                if fallback:
                    values["DEEPSEEK_API_KEY"] = fallback
        else:
            values = dict(environ)

        model = values.get("MODEL", "").strip()
        if not model:
            raise ValueError("MODEL is required")
        api_key = (values.get("API_KEY") or values.get("DEEPSEEK_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("API_KEY or DEEPSEEK_API_KEY is required")
        base_url = values.get("BASE_URL", "").strip()
        if not base_url:
            raise ValueError("BASE_URL is required")
        database_url = values.get("DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError("DATABASE_URL is required")

        try:
            timeout = float(values.get("TOOL_TIMEOUT_SECONDS", "5"))
            retries = int(values.get("TOOL_MAX_RETRIES", "2"))
        except ValueError as error:
            raise ValueError("Tool timeout and retry settings must be numeric") from error
        if timeout <= 0 or retries < 0:
            raise ValueError("Tool timeout must be positive and retries cannot be negative")

        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            database_url=database_url,
            tool_timeout_seconds=timeout,
            tool_max_retries=retries,
        )
