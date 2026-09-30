"""Extract a checkpointed batch of reusable support Q&A from chat history."""

from __future__ import annotations

import argparse

from app.services.knowledge.runtime import build_extractor as _build_extractor


def build_extractor():
    return _build_extractor()


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
    parser = argparse.ArgumentParser(description="Extract reusable knowledge from historical ecommerce support chats.")
    parser.add_argument("--batch-size", type=_positive_int, default=100)
    args = parser.parse_args(argv)
    indexer = None
    try:
        extractor, indexer = build_extractor()
        summary = extractor.run(batch_size=args.batch_size)
        pending, vectorized, failed = _sync_all(indexer, indexer.settings.knowledge_batch_size)
        print(
            f"conversation knowledge: messages={summary.messages_read} staged={summary.staged_pairs} "
            f"deduped={summary.deduped} inserted={summary.inserted} "
            f"skipped_tool_messages={summary.skipped_tool_messages} last_message_id={summary.last_message_id} "
            f"pending={pending} vectorized={vectorized} failed={failed}"
        )
        return 1 if failed else 0
    except Exception:
        print("conversation knowledge extraction failed: check configuration and service availability")
        return 1
    finally:
        if indexer is not None:
            indexer.close()


if __name__ == "__main__":
    raise SystemExit(main())
