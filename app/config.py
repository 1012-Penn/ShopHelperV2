"""Runtime configuration for the customer support demo."""

from __future__ import annotations

import os
import math
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
    embedding_api_key: str = ""
    embedding_api_base: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    milvus_uri: str = ""
    milvus_collection: str = "knowledge"
    knowledge_max_chars: int = 1200
    knowledge_overlap_chars: int = 200
    faq_top_k: int = 5
    faq_min_similarity: float = 0.60
    knowledge_batch_size: int = 32

    def require_knowledge(self) -> None:
        if not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY is required for knowledge features")
        if not self.milvus_uri:
            raise ValueError("MILVUS_URI is required for knowledge features")

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
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        require_chat: bool = True,
        allow_chat_key_fallback: bool = True,
    ) -> "Settings":
        if environ is None:
            values = {key: value for key, value in dotenv_values(Path.cwd() / ".env").items() if value is not None}
            values.update(os.environ)
            if (
                require_chat
                and allow_chat_key_fallback
                and not values.get("API_KEY")
                and not values.get("DEEPSEEK_API_KEY")
            ):
                fallback = dotenv_values("/root/.env").get("DEEPSEEK_API_KEY")
                if fallback:
                    values["DEEPSEEK_API_KEY"] = fallback
        else:
            values = dict(environ)

        model = values.get("MODEL", "").strip()
        if require_chat and not model:
            raise ValueError("MODEL is required")
        api_key = (values.get("API_KEY") or values.get("DEEPSEEK_API_KEY") or "").strip()
        if require_chat and not api_key:
            raise ValueError("API_KEY or DEEPSEEK_API_KEY is required")
        base_url = values.get("BASE_URL", "").strip()
        if require_chat and not base_url:
            raise ValueError("BASE_URL is required")
        database_url = values.get("DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError("DATABASE_URL is required")

        try:
            timeout = float(values.get("TOOL_TIMEOUT_SECONDS", "5"))
            retries = int(values.get("TOOL_MAX_RETRIES", "2"))
            max_chars = int(values.get("KNOWLEDGE_MAX_CHARS", "1200"))
            overlap_chars = int(values.get("KNOWLEDGE_OVERLAP_CHARS", "200"))
            top_k = int(values.get("FAQ_TOP_K", "5"))
            min_similarity = float(values.get("FAQ_MIN_SIMILARITY", "0.60"))
            batch_size = int(values.get("KNOWLEDGE_BATCH_SIZE", "32"))
        except ValueError as error:
            raise ValueError("Timeout, retry, chunk, retrieval, and batch settings must be numeric") from error
        if timeout <= 0 or retries < 0:
            raise ValueError("Tool timeout must be positive and retries cannot be negative")
        if max_chars <= overlap_chars or overlap_chars < 0:
            raise ValueError("KNOWLEDGE_MAX_CHARS must exceed non-negative KNOWLEDGE_OVERLAP_CHARS")
        if not 1 <= top_k <= 20:
            raise ValueError("FAQ_TOP_K must be between 1 and 20")
        if not math.isfinite(min_similarity) or not 0 <= min_similarity <= 1:
            raise ValueError("FAQ_MIN_SIMILARITY must be between 0 and 1")
        if batch_size < 1:
            raise ValueError("KNOWLEDGE_BATCH_SIZE must be positive")

        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            database_url=database_url,
            tool_timeout_seconds=timeout,
            tool_max_retries=retries,
            embedding_api_key=(values.get("EMBEDDING_API_KEY") or "").strip(),
            embedding_api_base=(
                values.get("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1").strip().rstrip("/")
            ),
            embedding_model=values.get("EMBEDDING_MODEL", "BAAI/bge-m3").strip(),
            milvus_uri=values.get("MILVUS_URI", "").strip(),
            milvus_collection=values.get("MILVUS_COLLECTION", "knowledge").strip(),
            knowledge_max_chars=max_chars,
            knowledge_overlap_chars=overlap_chars,
            faq_top_k=top_k,
            faq_min_similarity=min_similarity,
            knowledge_batch_size=batch_size,
        )
