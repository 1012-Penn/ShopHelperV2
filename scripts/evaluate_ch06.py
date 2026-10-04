"""Run the ch06 reference-resolution and intent prompts against labelled cases."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import Settings
from app.services.workflow.intents import (
    INTENT_PROMPT,
    parse_intent,
    resolve_question,
)
from app.services.workflow.demo_orders import get_demo_order, list_demo_orders
from app.services.workflow.query_expansion import expand_policy_queries


def _history(turns):
    result = []
    for turn in turns:
        if turn.get("role") == "user":
            result.append(HumanMessage(content=turn.get("content", "")))
        elif turn.get("role") == "assistant":
            result.append(AIMessage(content=turn.get("content", "")))
    return result


def evaluate_prompts(cases, model):
    rows = []
    for case in cases:
        row = {
            "id": case["id"],
            "conversation_id": case.get("conversation_id"),
            "input": case["input"],
            "expected": case["expected"],
        }
        try:
            resolution_model = _CaptureModel(model)
            question, reference_resolved = resolve_question(
                resolution_model,
                _history(case.get("turns", [])),
                case["input"],
            )
            row.update(
                resolved_query=question,
                reference_resolved=reference_resolved,
                reference_raw_json=resolution_model.last_content,
                reference_json_valid=True,
            )
        except Exception as error:  # preserve per-case model/parse failure for review
            row.update(
                resolved_query=None,
                reference_resolved=None,
                reference_raw_json=getattr(locals().get("resolution_model"), "last_content", None),
                reference_json_valid=False,
                reference_error=type(error).__name__,
            )
        try:
            intent_response = model.invoke(
                [
                    SystemMessage(content=INTENT_PROMPT),
                    HumanMessage(content=row.get("resolved_query") or case["input"]),
                ]
            )
            row["intent_raw_json"] = intent_response.content
            intent, confidence = parse_intent(intent_response.content)
            row.update(intent=intent, confidence=confidence, intent_json_valid=True)
        except Exception as error:  # preserve the raw provider output when parsing fails
            row.update(
                intent=None,
                confidence=None,
                intent_json_valid=False,
                intent_error=type(error).__name__,
            )
        row["intent_correct"] = row.get("intent") == case["expected"].get("intent")
        row["query_correct"] = row.get("resolved_query") == case["expected"].get(
            "resolved_query"
        )
        row["reference_status_correct"] = row.get("reference_resolved") is case[
            "expected"
        ].get("reference_resolved")
        rows.append(row)
    return rows


class _CaptureModel:
    def __init__(self, model):
        self.model = model
        self.last_content = None

    def invoke(self, messages):
        result = self.model.invoke(messages)
        self.last_content = result.content
        return result


def evaluate_expansion(cases, model):
    rows = []
    for case in cases:
        focus = case.get("expansion_focus")
        if not focus:
            continue
        question = case["expected"]["resolved_query"]
        intent = case["expected"]["intent"]
        order_id = re.search(r"DEMO-\d+", question)
        order = get_demo_order(
            order_id.group(0) if order_id else list_demo_orders("demo-user")[0]["order_id"],
            "demo-user",
        )
        capturing = _CaptureModel(model)
        row = {
            "id": case["id"],
            "intent": intent,
            "resolved_question": question,
            "expected_focus": focus,
        }
        try:
            queries = expand_policy_queries(capturing, question, order, intent)
            raw = capturing.last_content
            try:
                parsed = json.loads(raw)
                valid = (
                    isinstance(parsed, dict)
                    and set(parsed) == {"queries"}
                    and isinstance(parsed["queries"], list)
                    and 1 <= len(parsed["queries"]) <= 3
                    and all(
                        isinstance(item, str)
                        and item.strip()
                        and len(item) <= 240
                        for item in parsed["queries"]
                    )
                    and len({item.strip().casefold() for item in parsed["queries"]})
                    == len(parsed["queries"])
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                valid = False
            row.update(
                raw_json=raw,
                queries=queries,
                json_valid=valid,
                distinct_queries=len({item.casefold() for item in queries}) == len(queries),
                fallback_included=question in queries,
            )
        except Exception as error:
            row.update(
                raw_json=capturing.last_content,
                queries=[question],
                json_valid=False,
                distinct_queries=True,
                fallback_included=True,
                error=type(error).__name__,
            )
        rows.append(row)
    return rows


def summarize(rows):
    count = len(rows)
    denominator = max(count, 1)
    return {
        "case_count": count,
        "intent_json_valid": sum(row["intent_json_valid"] for row in rows),
        "intent_accuracy": sum(row["intent_correct"] for row in rows) / denominator,
        "reference_json_valid": sum(row["reference_json_valid"] for row in rows),
        "reference_query_exact": sum(row["query_correct"] for row in rows) / denominator,
        "reference_status_accuracy": sum(
            row["reference_status_correct"] for row in rows
        )
        / denominator,
    }


def summarize_expansion(rows):
    count = len(rows)
    denominator = max(count, 1)
    return {
        "case_count": count,
        "query_json_valid": sum(row["json_valid"] for row in rows),
        "query_json_valid_rate": sum(row["json_valid"] for row in rows) / denominator,
        "distinct_query_rate": sum(row["distinct_queries"] for row in rows) / denominator,
        "fallback_included": sum(row["fallback_included"] for row in rows),
        "mean_queries_including_fallback": sum(len(row["queries"]) for row in rows)
        / denominator,
    }


def _write_results(output_dir, metadata, rows, *, expansion_rows=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if metadata["stage"] == "expansion":
        metrics = summarize_expansion(rows)
        report = [
            "# ch06 Query Expansion Evaluation",
            "",
            f"- Model: `{metadata.get('model', 'unavailable')}`",
            f"- Run started: `{metadata['started_at']}`",
            f"- High-risk cases: {metrics['case_count']}",
            f"- Expansion JSON contract valid: {metrics['query_json_valid']}/{metrics['case_count']}",
            f"- Distinct-query cases: {metrics['distinct_query_rate']:.1%}",
            f"- Canonical fallback included: {metrics['fallback_included']}/{metrics['case_count']}",
            f"- Mean queries including fallback: {metrics['mean_queries_including_fallback']:.2f}",
        ]
    else:
        metrics = summarize(rows)
        report = [
            "# ch06 Prompt Evaluation",
            "",
            f"- Model: `{metadata.get('model', 'unavailable')}`",
            f"- Run started: `{metadata['started_at']}`",
            f"- Cases: {metrics['case_count']}",
            f"- Intent JSON valid: {metrics['intent_json_valid']}/{metrics['case_count']}",
            f"- Intent accuracy: {metrics['intent_accuracy']:.1%}",
            f"- Reference JSON valid: {metrics['reference_json_valid']}/{metrics['case_count']}",
            f"- Exact resolved query: {metrics['reference_query_exact']:.1%}",
            f"- Reference-resolved flag accuracy: {metrics['reference_status_accuracy']:.1%}",
            "- Exact query match is a strict string metric; natural paraphrases require semantic review.",
        ]
    if expansion_rows is not None:
        expansion_metrics = summarize_expansion(expansion_rows)
        with (output_dir / "results-expansion.jsonl").open("w", encoding="utf-8") as stream:
            for row in expansion_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report.extend(
            [
                "",
                "## Query Expansion",
                f"- High-risk cases: {expansion_metrics['case_count']}",
                f"- Expansion JSON contract valid: {expansion_metrics['query_json_valid']}/{expansion_metrics['case_count']}",
                f"- Distinct-query cases: {expansion_metrics['distinct_query_rate']:.1%}",
                f"- Mean queries including fallback: {expansion_metrics['mean_queries_including_fallback']:.2f}",
            ]
        )
    if metadata.get("run_error"):
        report.extend(["", f"Run error: `{metadata['run_error']}`"])
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prompts", "expansion", "all"), default="all")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    metadata = {
        "stage": args.stage,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "case_file": str(args.cases),
    }
    rows = []
    expansion_rows = None
    try:
        settings = Settings.from_env()
        from langchain_openai import ChatOpenAI

        metadata["model"] = settings.model
        model = ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=0,
            timeout=45,
            max_retries=0,
        )
        if args.stage in {"prompts", "all"}:
            rows = evaluate_prompts(cases, model)
        if args.stage in {"expansion", "all"}:
            evaluated_expansion = evaluate_expansion(cases, model)
            if args.stage == "expansion":
                rows = evaluated_expansion
            else:
                expansion_rows = evaluated_expansion
    except Exception as error:
        # Do not serialize settings or provider exception text; both can contain secrets.
        metadata["run_error"] = type(error).__name__
        _write_results(args.output_dir, metadata, rows, expansion_rows=expansion_rows)
        print(f"Evaluation unavailable ({type(error).__name__}); report saved to {args.output_dir}")
        return 2
    metrics = _write_results(args.output_dir, metadata, rows, expansion_rows=expansion_rows)
    if expansion_rows is not None:
        metrics = {"prompts": metrics, "expansion": summarize_expansion(expansion_rows)}
    print(json.dumps(metrics, ensure_ascii=False))
    print(f"Report saved to {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
