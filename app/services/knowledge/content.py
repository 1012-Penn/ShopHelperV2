"""Canonical knowledge draft and embedding-text helpers."""

from dataclasses import dataclass


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
