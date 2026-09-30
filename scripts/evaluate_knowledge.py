"""Evaluate dense FAQ retrieval against a small labeled ecommerce support set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from app.services.knowledge.retriever import DenseSearcher
from app.services.knowledge.runtime import build_indexer


def evaluate_cases(cases: list[dict[str, Any]], searcher: Any, repository: Any, thresholds: list[float]) -> dict[str, Any]:
    """Measure labeled source/answer hits and unrelated-query false positives."""
    if not cases or not thresholds:
        raise ValueError("evaluation cases and thresholds must not be empty")
    if any(not 0 <= threshold <= 1 for threshold in thresholds):
        raise ValueError("thresholds must be between 0 and 1")

    evaluated = []
    for case in cases:
        query = str(case["query"])
        hits = searcher.search(query, limit=5)
        rows = repository.load_by_ids([hit.chunk_id for hit in hits])
        rows_by_id = {row.id: row for row in rows}
        evaluated.append((case, [(hit, rows_by_id.get(hit.chunk_id)) for hit in hits]))

    metrics: dict[float, dict[str, int | float]] = {}
    positives = sum(case["expected"] == "hit" for case, _ in evaluated)
    negatives = sum(case["expected"] == "miss" for case, _ in evaluated)
    for threshold in sorted(set(thresholds)):
        correct = 0
        false_positives = 0
        for case, hits in evaluated:
            accepted = [(hit, row) for hit, row in hits if hit.score >= threshold and row is not None]
            if case["expected"] == "miss":
                false_positives += bool(accepted)
                continue
            for _, row in accepted:
                expected_source = PurePosixPath(case["source"]).name
                source_matches = expected_source in row.source_key
                answer_matches = case["answer_contains"] in row.answer
                if source_matches and answer_matches:
                    correct += 1
                    break
        metrics[threshold] = {
            "answer_source_correct": correct,
            "positive_cases": positives,
            "positive_recall": correct / positives if positives else 0.0,
            "negative_false_positives": false_positives,
            "negative_cases": negatives,
            "negative_false_positive_rate": false_positives / negatives if negatives else 0.0,
        }

    qualified = [
        threshold
        for threshold, values in metrics.items()
        if values["answer_source_correct"] == positives and values["negative_false_positives"] == 0
    ]
    return {"metrics": metrics, "selected_threshold": max(qualified) if qualified else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate FAQ dense retrieval on labeled ecommerce support cases")
    parser.add_argument("--cases", type=Path, default=Path("tests/fixtures/faq_cases.json"))
    parser.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    args = parser.parse_args(argv)

    indexer = None
    try:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
        indexer = build_indexer()
        searcher = DenseSearcher(indexer.embeddings, indexer.vector_store)
        result = evaluate_cases(cases, searcher, indexer.repository, thresholds)
        for threshold, values in result["metrics"].items():
            print(
                f"threshold={threshold:.2f} answer_source_correct={values['answer_source_correct']}/"
                f"{values['positive_cases']} negative_false_positives={values['negative_false_positives']}/"
                f"{values['negative_cases']}"
            )
        if result["selected_threshold"] is None:
            print("No threshold met all labeled positive and negative cases; review the corpus or labels.")
            return 1
        print(f"selected_threshold={result['selected_threshold']:.2f}")
        return 0
    except Exception:
        print("Knowledge evaluation failed; check configuration, data, and service availability.")
        return 1
    finally:
        if indexer is not None:
            indexer.close()


if __name__ == "__main__":
    raise SystemExit(main())
