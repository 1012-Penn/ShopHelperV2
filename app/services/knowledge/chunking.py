"""Structure-aware Markdown chunking for ecommerce support knowledge."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from app.services.knowledge.content import ChunkDraft

_SENTENCE_END = re.compile(r"(?<=[。！？.!?])\s*")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_CRITICAL = re.compile(r"退款|退货|退换|支付安全|账户安全|隐私|发票|保修|质保|取消订单")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split(text.strip()) if part.strip()]


def _split_prose(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    sentences = _sentences(text)
    if not sentences:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        overlap: list[str] = []
        overlap_size = 0
        if chunks and overlap_chars:
            previous_sentences = _sentences(chunks[-1])
            for sentence in reversed(previous_sentences):
                addition = len(sentence) + (1 if overlap else 0)
                if overlap and overlap_size + addition > overlap_chars:
                    break
                overlap.insert(0, sentence)
                overlap_size += addition
        if overlap and len(" ".join(overlap + [sentences[start]])) > max_chars:
            overlap = []

        current = list(overlap)
        current_size = len(" ".join(current))
        end = start
        while end < len(sentences):
            sentence = sentences[end]
            candidate_size = current_size + (1 if current else 0) + len(sentence)
            if current and candidate_size > max_chars:
                break
            current.append(sentence)
            current_size = candidate_size
            end += 1
        # A single overlong sentence is intentionally retained as a whole.
        chunks.append(" ".join(current))
        start = end
    return chunks


def _split_table(lines: list[str], max_chars: int) -> list[str]:
    if len(lines) <= 2:
        return ["\n".join(lines)]
    header = lines[:2]
    rows = lines[2:]
    result: list[str] = []
    current: list[str] = []
    for row in rows:
        candidate = header + current + [row]
        if current and len("\n".join(candidate)) > max_chars:
            result.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        result.append("\n".join(header + current))
    return result or ["\n".join(lines)]


def _content_type(source: str, path: list[str]) -> str:
    value = (source + " " + " ".join(path)).lower()
    if any(word in value for word in ("faq", "常见问题")):
        return "product_faq"
    if any(word in value for word in ("售后", "维修", "质保", "保修")):
        return "after_sales"
    return "policy"


def split_markdown(
    source: str,
    markdown: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[ChunkDraft]:
    """Split Markdown by headings, paragraphs, tables, sentences, and code fences.

    Character limits are soft: one sentence, table row, or fenced code block is
    never truncated to satisfy the target.
    """
    if max_chars < 1 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("max_chars must be positive and overlap_chars must be in [0, max_chars)")

    headings: list[str] = []
    segments: list[tuple[list[str], str, str]] = []
    paragraph: list[str] = []
    lines = markdown.splitlines()
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            segments.append((headings.copy(), "prose", " ".join(line.strip() for line in paragraph).strip()))
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        heading_match = _HEADING.match(line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            headings[:] = headings[: level - 1]
            headings.append(title)
            index += 1
            continue
        if not line.strip():
            flush_paragraph()
            index += 1
            continue
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            flush_paragraph()
            marker = "```" if line.lstrip().startswith("```") else "~~~"
            code = [line]
            index += 1
            while index < len(lines):
                code.append(lines[index])
                is_end = lines[index].lstrip().startswith(marker)
                index += 1
                if is_end:
                    break
            segments.append((headings.copy(), "code", "\n".join(code)))
            continue
        if line.lstrip().startswith("|"):
            flush_paragraph()
            table = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table.append(lines[index].strip())
                index += 1
            segments.append((headings.copy(), "table", "\n".join(table)))
            continue
        paragraph.append(line)
        index += 1
    flush_paragraph()

    merged_segments: list[tuple[list[str], str, str]] = []
    for path, kind, content in segments:
        if (
            kind == "prose"
            and merged_segments
            and merged_segments[-1][0] == path
            and merged_segments[-1][1] == "prose"
        ):
            previous_path, previous_kind, previous_content = merged_segments[-1]
            merged_segments[-1] = (previous_path, previous_kind, f"{previous_content} {content}")
        else:
            merged_segments.append((path, kind, content))

    drafts: list[ChunkDraft] = []
    path_counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
    current_question: str | None = None
    for path, kind, content in merged_segments:
        content = content.strip()
        if not content:
            continue
        if path and path[-1].endswith(("?", "？")):
            current_question = path[-1]
        parts = (
            _split_table(content.splitlines(), max_chars)
            if kind == "table"
            else [content]
            if kind == "code"
            else _split_prose(content, max_chars, overlap_chars)
        )
        for part in parts:
            if not part.strip():
                continue
            parent_path = path[:-1] if len(path) > 1 else []
            category = " / ".join(parent_path) if parent_path else "电商客服"
            question = current_question or (path[-1] if path else "电商客服知识")
            ordinal = path_counts[tuple(path)]
            path_counts[tuple(path)] += 1
            digest = hashlib.sha256(f"{source}\0{'/'.join(path)}\0{ordinal}".encode()).hexdigest()[:16]
            drafts.append(
                ChunkDraft(
                    source_key=f"doc:{source}:{digest}",
                    category=category,
                    questions=[question],
                    answer=part,
                    chapter_path=path.copy(),
                    content_type=_content_type(source, path),
                    is_critical=bool(_CRITICAL.search(" ".join(path) + " " + part[:160])),
                )
            )
    for index, draft in enumerate(drafts):
        drafts[index] = ChunkDraft(
            source_key=draft.source_key,
            category=draft.category,
            questions=draft.questions,
            answer=draft.answer,
            chapter_path=draft.chapter_path,
            content_type=draft.content_type,
            is_critical=draft.is_critical,
            previous_source_key=drafts[index - 1].source_key if index else None,
            next_source_key=drafts[index + 1].source_key if index + 1 < len(drafts) else None,
        )
    return drafts
