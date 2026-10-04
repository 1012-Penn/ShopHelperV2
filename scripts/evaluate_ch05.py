"""Labelled intent prompt evaluation; fixture output never represents live quality."""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.services.workflow.intents import (
    INTENT_CONFIDENCE_THRESHOLD,
    INTENT_PROMPT,
    ROUTES,
    parse_intent,
)


def summarize(rows):
    correct = sum(not r.get("error") and r.get("actual") == r["expected"] for r in rows)
    return {
        "total": len(rows),
        "correct": correct,
        "errors": sum(bool(r.get("error")) for r in rows),
        "accuracy": correct / len(rows) if rows else 0,
    }


def evaluate(live=False, output_dir=None):
    cases = json.loads(Path("evaluation/ch05/cases.json").read_text())
    model_identity = "scripted fixture"
    if live:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        from app.config import Settings

        settings = Settings.from_env()
        model_identity = settings.model
        model = ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=0,
            timeout=45,
            max_retries=0,
            max_tokens=128,
        )

    def one(case):
        row = {
            **case,
            "actual": None,
            "raw": None,
            "error": None,
            "model_calls": 0,
            "usage": None,
        }
        if not live:
            row.update(actual=case["expected"], raw="synthetic fixture label")
        else:
            row["model_calls"] = 1
            try:
                response = model.invoke(
                    [
                        SystemMessage(content=INTENT_PROMPT),
                        HumanMessage(content=case["query"]),
                    ]
                )
                row.update(raw=response.content, usage=response.usage_metadata)
                try:
                    intent, confidence = parse_intent(response.content)
                    row["actual"] = (
                        intent
                        if confidence >= INTENT_CONFIDENCE_THRESHOLD
                        else "其他"
                    )
                except (ValueError, TypeError):
                    row["error"] = "invalid_json_or_intent"
            except Exception as error:  # noqa: BLE001 - isolate external provider/tool failures
                logging.getLogger(__name__).warning(
                    "Intent evaluation failed (%s)", type(error).__name__
                )
                row["error"] = type(error).__name__
        row["route"] = ROUTES.get(row["actual"])
        return row

    with ThreadPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(one, cases))
    report = {
        "mode": "live" if live else "fixture",
        "synthetic": not live,
        "model": model_identity,
        "summary": summarize(rows),
        "cases": rows,
    }
    out = Path(
        output_dir
        or (
            "evaluation/ch05/runs/"
            + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid4().hex[:6]
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "intents.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    s = report["summary"]
    (out / "intents.md").write_text(
        f"# ch05 intent evaluation ({report['mode']})\n\n"
        f"Model: {model_identity}. Correct {s['correct']}/{s['total']}; protocol/API errors {s['errors']}.\n\n"
        "28 labelled examples are a chapter baseline, not a production intent benchmark. Fixture is synthetic.\n\n"
        "| ID | Query | Expected | Actual | Error |\n|---|---|---|---|---|\n"
        + "\n".join(
            f"| {r['id']} | {r['query']} | {r['expected']} | {r['actual']} | {r['error'] or ''} |"
            for r in rows
        )
        + "\n"
    )
    return report, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    report, out = evaluate(args.live, args.output_dir)
    print(
        json.dumps(
            {"mode": report["mode"], **report["summary"], "output": str(out)},
            ensure_ascii=False,
        )
    )
    if report["summary"]["correct"] != report["summary"]["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
