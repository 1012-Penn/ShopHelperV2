"""Retry MySQL knowledge rows whose vector status is still pending."""

from __future__ import annotations

import argparse

from app.services.knowledge.runtime import build_indexer as _build_indexer


def build_indexer():
    return _build_indexer()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair pending dense knowledge vectors.")
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    args = parser.parse_args(argv)
    indexer = None
    try:
        indexer = build_indexer()
        summary = indexer.sync_pending(batch_size=args.batch_size)
        print(
            f"pending vector sync: pending={summary.pending_before} "
            f"vectorized={summary.vectorized} failed={summary.failed}"
        )
        return 1 if summary.failed else 0
    except Exception:
        print("pending vector sync failed: check configuration and service availability")
        return 1
    finally:
        if indexer is not None:
            indexer.close()


if __name__ == "__main__":
    raise SystemExit(main())
