import argparse
import json
from pathlib import Path
from typing import Any

from app.services.after_sale import AfterSaleService


class FixtureStructuredModel:
    def __init__(self, expected_by_text: dict[str, dict[str, Any]]):
        self.expected_by_text = expected_by_text

    def invoke(self, messages):
        text = messages[-1].content
        return self.expected_by_text[text]


def load_samples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def evaluate_fixture(samples: list[dict[str, Any]]) -> tuple[int, int, int]:
    model = FixtureStructuredModel(
        {sample["text"]: sample["expected"] for sample in samples}
    )
    service = AfterSaleService(model)
    passed = 0
    field_matches = 0
    field_total = len(samples) * 3
    for sample in samples:
        result = service.extract(sample["text"]).model_dump()
        expected = sample["expected"]
        matches = sum(result[field] == expected[field] for field in expected)
        field_matches += matches
        if result == expected:
            passed += 1
    return passed, len(samples), field_matches * 100 // field_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("evals/after_sale_samples.jsonl"),
    )
    args = parser.parse_args()
    if not args.fixture:
        parser.error("本章只提供 --fixture 离线评估；真实模型评估需显式接入上游")

    samples = load_samples(args.samples)
    passed, total, accuracy = evaluate_fixture(samples)
    print(f"samples={total} passed={passed} failed={total - passed} field_accuracy={accuracy}%")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
