"""Canonical knowledge draft and embedding-text helpers."""

from dataclasses import dataclass
import hashlib
import re
import unicodedata


@dataclass(frozen=True)
class ChunkDraft:
    source_key: str
    category: str
    questions: list[str]
    answer: str
    chapter_path: list[str]
    content_type: str
    is_critical: bool
    previous_source_key: str | None = None
    next_source_key: str | None = None


def build_embedding_text(category: str, questions: list[str], answer: str) -> str:
    """Serialize only retrieval-bearing fields in a stable, line-oriented form."""
    return "\n".join([category.strip(), *(question.strip() for question in questions), answer.strip()])


def build_content_hash(category: str, questions: list[str], answer: str) -> str:
    """Hash exact normalized retrieval content to decide whether vectors are stale."""
    content = unicodedata.normalize("NFC", build_embedding_text(category, questions, answer))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_knowledge_fingerprint(category: str, questions: list[str], answer: str) -> str:
    """Hash normalized text for exact duplicate detection across knowledge sources."""
    value = unicodedata.normalize("NFKC", build_embedding_text(category, questions, answer)).casefold()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[\W_]+", "", value, flags=re.UNICODE)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
