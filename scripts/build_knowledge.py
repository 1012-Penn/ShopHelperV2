"""Import Markdown and legacy FAQ rows, then vectorize all pending chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.services.knowledge.chunking import split_markdown
from app.services.knowledge.runtime import build_indexer as _build_indexer


def build_indexer():
    return _build_indexer()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _sync_all(indexer, batch_size: int) -> tuple[int, int, int]:
    pending = vectorized = failed = 0
    while True:
        summary = indexer.sync_pending(batch_size=batch_size)
        pending += summary.pending_before
        vectorized += summary.vectorized
        failed += summary.failed
        if summary.failed or summary.pending_before < batch_size or summary.vectorized == 0:
            break
    return pending, vectorized, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ecommerce support knowledge from Markdown and legacy FAQ.")
    parser.add_argument("--source-dir", default="knowledge_docs")
    parser.add_argument("--batch-size", type=_positive_int)
    args = parser.parse_args(argv)
    indexer = None
    try:
        indexer = build_indexer()
        batch_size = args.batch_size or indexer.settings.knowledge_batch_size
        source_dir = Path(args.source_dir)
        if not source_dir.is_dir():
            print("knowledge build failed: source directory is unavailable")
            return 1
        documents = sorted(source_dir.glob("*.md"))
        if not documents:
            print("knowledge build failed: source directory contains no Markdown documents")
            return 1
        drafts = []
        for path in documents:
            drafts.extend(
                split_markdown(
                    path.name,
                    path.read_text(encoding="utf-8"),
                    max_chars=indexer.settings.knowledge_max_chars,
                    overlap_chars=indexer.settings.knowledge_overlap_chars,
                )
            )
        chunk_ids = indexer.import_markdown(drafts, [path.name for path in documents])
        faq_ids = indexer.import_faqs()
        pending, vectorized, failed = _sync_all(indexer, batch_size)
        print(
            f"knowledge build: documents={len(documents)} chunks={len(chunk_ids)} "
            f"legacy_faqs={len(faq_ids)} pending={pending} vectorized={vectorized} failed={failed}"
        )
        return 1 if failed else 0
    except Exception:
        print("knowledge build failed: check configuration and service availability")
        return 1
    finally:
        if indexer is not None:
            indexer.close()


if __name__ == "__main__":
    raise SystemExit(main())
