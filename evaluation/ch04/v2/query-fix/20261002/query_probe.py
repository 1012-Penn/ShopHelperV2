#!/usr/bin/env python3
"""Capture production query normalization on frozen calibration inputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "evaluation/ch04/v2"
OUT = DATA / "query-fix/20261002"
INITIAL_IDS = ("V2A003", "V2B001", "V2C001", "V2C003")
MAX_CALLS = 60


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    return str(value)


def read_calibration():
    cases_path = DATA / "cases.json"
    all_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    calibration = [case for case in all_cases if case.get("split") == "calibration"]
    if len(all_cases) != 300 or len(calibration) != MAX_CALLS:
        raise RuntimeError(f"expected frozen 300-case dataset with 60 calibration rows; got {len(all_cases)}/{len(calibration)}")
    if any(case.get("split") != "calibration" for case in calibration):
        raise RuntimeError("refusing to send any test split row")
    by_id = {case["eval_id"]: case for case in calibration}
    if any(eval_id not in by_id for eval_id in INITIAL_IDS):
        raise RuntimeError("one or more required initial cases are missing from calibration")
    from scripts.validate_ch04_dataset import load_corpus, verify_frozen_dataset
    verify_frozen_dataset(DATA, load_corpus(DATA / "corpus"))
    return all_cases, calibration, by_id, sha256_bytes(cases_path.read_bytes())


def selected_cases(stage: str, calibration: list[dict], by_id: dict[str, dict]):
    if stage == "initial":
        return [by_id[eval_id] for eval_id in INITIAL_IDS]
    return [case for case in calibration if case["eval_id"] not in INITIAL_IDS]


def message_dump(messages):
    output = []
    for message in messages:
        if isinstance(message, dict):
            output.append(json_safe(message))
        elif hasattr(message, "model_dump"):
            output.append(json_safe(message.model_dump()))
        else:
            output.append({"type": type(message).__name__, "repr": str(message)})
    return output


class CapturingRunnable:
    def __init__(self, runnable, captures):
        self.runnable = runnable
        self.captures = captures

    def invoke(self, messages):
        entry = {"messages": message_dump(messages)}
        try:
            response = self.runnable.invoke(messages)
        except Exception as error:
            entry.update({"request_error_type": type(error).__name__})
            self.captures.append(entry)
            raise
        raw = response.get("raw")
        parsed = response.get("parsed")
        parsing_error = response.get("parsing_error")
        entry.update({
            "raw_content": json_safe(getattr(raw, "content", None)),
            "raw_response_metadata": json_safe(getattr(raw, "response_metadata", None)),
            "raw_usage_metadata": json_safe(getattr(raw, "usage_metadata", None)),
            "parsed": parsed.model_dump() if parsed is not None else None,
            "parsing_error_type": type(parsing_error).__name__ if parsing_error else None,
            "parsing_error": str(parsing_error) if parsing_error else None,
        })
        self.captures.append(entry)
        # QueryNormalizer.from_model receives the same parsed Pydantic object
        # that it would receive without this capture adapter.
        return parsed


class CapturingChatModel:
    def __init__(self, model, captures):
        self.model = model
        self.captures = captures

    def with_structured_output(self, schema, *, method):
        runnable = self.model.with_structured_output(
            schema, method=method, include_raw=True)
        return CapturingRunnable(runnable, self.captures)


def guard_record(raw: str, candidate: str):
    from app.services.quality.query import (
        _identifier_signature,
        _negative_clauses,
        _quantity_signature,
        _unknown_quantity_clauses,
    )

    raw_ids, candidate_ids = _identifier_signature(raw), _identifier_signature(candidate)
    raw_quantities, candidate_quantities = _quantity_signature(raw), _quantity_signature(candidate)
    raw_unknown, candidate_unknown = _unknown_quantity_clauses(raw), _unknown_quantity_clauses(candidate)
    raw_negative, candidate_negative = _negative_clauses(raw), _negative_clauses(candidate)
    failed = []
    if not candidate.strip():
        failed.append("empty_candidate")
    if raw_ids != candidate_ids:
        failed.append("identifier_sequence_mismatch")
    if raw_quantities != candidate_quantities:
        failed.append("quantity_or_unit_mismatch")
    if raw_unknown != candidate_unknown:
        failed.append("unknown_unit_clause_mismatch")
    if raw_negative != candidate_negative:
        failed.append("negative_scope_mismatch")
    return {
        "accepted": bool(candidate.strip()) and not failed,
        "failed_predicates": failed,
        "raw": {"identifiers": raw_ids, "quantities": raw_quantities,
                "unknown_quantity_clauses": raw_unknown, "negative_clauses": raw_negative},
        "candidate": {"identifiers": candidate_ids, "quantities": candidate_quantities,
                      "unknown_quantity_clauses": candidate_unknown, "negative_clauses": candidate_negative},
    }


def classify(raw: str, candidate: str, canonical: str, downgrade_reason: str):
    if canonical != raw.strip():
        return "changedAccepted"
    if candidate.strip() == raw.strip() and not downgrade_reason:
        return "unchangedAccepted"
    return "fallback"


def make_item(case: dict, capture: dict, understanding):
    parsed = capture.get("parsed")
    candidate = parsed.get("canonical", "") if parsed else ""
    messages = capture.get("messages") or []
    system_prompt = next((message.get("content", "") for message in messages
                          if message.get("role") == "system"), "")
    from app.services.quality.query import NormalizedQuery
    schema = NormalizedQuery.model_json_schema()
    return {
        "eval_id": case["eval_id"],
        "bucket": case["bucket"],
        "split": case["split"],
        "query": case["query"],
        "candidate": candidate,
        "canonical": understanding.canonical,
        "downgrade_reason": understanding.downgrade_reason,
        "outcome": classify(case["query"], candidate, understanding.canonical,
                             understanding.downgrade_reason),
        "guard": guard_record(case["query"], candidate),
        "requires_human_semantic_review": understanding.canonical != case["query"].strip(),
        "captured_messages": messages,
        "prompt_sha256": sha256_bytes(system_prompt.encode("utf-8")),
        "schema_sha256": digest_json(schema),
        "method": "json_mode",
        "include_raw": True,
        "raw_content": capture.get("raw_content"),
        "raw_response_metadata": capture.get("raw_response_metadata"),
        "raw_usage_metadata": capture.get("raw_usage_metadata"),
        "parsed": parsed,
        "parsing_error_type": capture.get("parsing_error_type"),
        "parsing_error": capture.get("parsing_error"),
        "request_error_type": capture.get("request_error_type"),
    }


def dump_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def run(stage: str):
    all_cases, calibration, by_id, case_file_hash = read_calibration()
    cases = selected_cases(stage, calibration, by_id)
    if stage == "initial":
        if len(cases) != 4:
            raise RuntimeError("initial stage must be exactly four cases")
        if (OUT / "run.json").exists() or (OUT / "initial.jsonl").exists():
            raise RuntimeError("initial outputs already exist; refusing duplicate live calls")
    else:
        initial_path = OUT / "initial.jsonl"
        if not initial_path.exists() or sum(1 for line in initial_path.read_text().splitlines() if line.strip()) != 4:
            raise RuntimeError("remaining stage requires exactly four saved initial results")
        if (OUT / "remaining.jsonl").exists():
            raise RuntimeError("remaining outputs already exist; refusing duplicate live calls")
        if len(cases) != 56:
            raise RuntimeError("remaining stage must be exactly 56 calibration cases")

    from app.config import Settings
    from app.services.quality.query import NormalizedQuery, QueryNormalizer
    from langchain_openai import ChatOpenAI

    settings = Settings.from_env()
    if not settings.model or not settings.api_key or not settings.base_url:
        raise RuntimeError("chat model settings are incomplete")
    model = ChatOpenAI(model=settings.model, api_key=settings.api_key,
                       base_url=settings.base_url, temperature=0, timeout=90,
                       max_retries=0)
    captures = []
    normalizer = QueryNormalizer.from_model(CapturingChatModel(model, captures))
    output_path = OUT / ("initial.jsonl" if stage == "initial" else "remaining.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)

    if stage == "initial":
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "model": settings.model,
            "endpoint_host": urlsplit(settings.base_url).hostname,
            "temperature": 0,
            "max_retries": 0,
            "method": "json_mode",
            "include_raw": True,
            "schema": NormalizedQuery.model_json_schema(),
            "calibration_count": len(calibration),
            "total_live_call_budget": MAX_CALLS,
            "inputs_sha256": case_file_hash,
            "calibration_sha256": digest_json(calibration),
            "initial_eval_ids": list(INITIAL_IDS),
            "excluded_test_cases": sum(1 for case in all_cases if case.get("split") == "test"),
            "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        }
        with (OUT / "run.json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    total_previous = 0 if stage == "initial" else 4
    if total_previous + len(cases) > MAX_CALLS:
        raise RuntimeError("refusing to exceed the 60-call calibration budget")
    with output_path.open("x", encoding="utf-8") as handle:
        for index, case in enumerate(cases, 1):
            before = len(captures)
            understanding = normalizer.normalize(case["query"])
            if len(captures) != before + 1:
                raise RuntimeError(f"expected one API capture for {case['eval_id']}, got {len(captures) - before}")
            item = make_item(case, captures[-1], understanding)
            item["stage_index"] = index
            item["total_call_number"] = total_previous + index
            dump_line(handle, item)
            print(json.dumps({"eval_id": item["eval_id"], "outcome": item["outcome"],
                              "canonical": item["canonical"], "downgrade_reason": item["downgrade_reason"],
                              "failed_predicates": item["guard"]["failed_predicates"]},
                             ensure_ascii=False), flush=True)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    run_path = OUT / "run.json"
    if stage == "initial":
        run_metadata = json.loads(run_path.read_text(encoding="utf-8"))
        prompts = [next((message.get("content", "") for message in row["captured_messages"]
                         if message.get("role") == "system"), "") for row in rows]
        if prompts:
            run_metadata["system_prompt"] = prompts[0]
            run_metadata["prompt_sha256"] = sha256_bytes(prompts[0].encode("utf-8"))
            run_metadata["schema_sha256"] = digest_json(run_metadata["schema"])
            if any(prompt != prompts[0] for prompt in prompts):
                raise RuntimeError("production prompt changed during the initial capture")
            run_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        remaining_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in remaining_rows:
            if row["prompt_sha256"] != previous.get("prompt_sha256"):
                raise RuntimeError("production prompt differs from initial capture")
            if row["schema_sha256"] != previous.get("schema_sha256"):
                raise RuntimeError("structured output schema differs from initial capture")

    all_rows = rows
    if stage == "remaining":
        all_rows = [json.loads(line) for line in (OUT / "initial.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] + rows
    counts = {name: sum(row["outcome"] == name for row in rows)
              for name in ("changedAccepted", "unchangedAccepted", "fallback")}
    all_counts = {name: sum(row["outcome"] == name for row in all_rows)
                  for name in ("changedAccepted", "unchangedAccepted", "fallback")}
    summary = {"stage": stage, "stage_calls": len(rows), "total_calibration_calls": len(all_rows),
               "stage_outcomes": counts, "outcomes": all_counts,
               "fallback_ids": [row["eval_id"] for row in all_rows if row["outcome"] == "fallback"],
               "changed_ids_pending_human_review": [row["eval_id"] for row in all_rows if row["requires_human_semantic_review"]],
               "note": "Guard acceptance checks preservation predicates only; semantic equivalence requires human review."}
    summary_path = OUT / ("initial-summary.json" if stage == "initial" else "calibration60-summary.json")
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if stage == "remaining":
        report = [
            "# Query normalization calibration capture (2026-10-02)",
            "",
            f"Total calls: {len(all_rows)}/60, all from frozen `calibration`; test rows: 0.",
            f"Outcomes: changedAccepted={all_counts['changedAccepted']}, unchangedAccepted={all_counts['unchangedAccepted']}, fallback={all_counts['fallback']}.",
            "",
            "A `changedAccepted` result passed structural preservation guards; it is not a semantic-equivalence verdict. Review those rows against the original intent before treating them as good rewrites. `unchangedAccepted` means the model returned the original wording and is not counted as a successful rewrite. Fallback includes rejected, empty, unparsable, or failed model responses.",
            "",
            f"Changed candidates pending human review: {', '.join(summary['changed_ids_pending_human_review']) or 'none'}.",
            f"Fallback cases: {', '.join(summary['fallback_ids']) or 'none'}.",
            "",
            "The guard recognizes common quantities and units and locks an atomic clause when a number is followed by an unrecognized Chinese unit. This is a conservative boundary, not a complete Chinese quantity or semantic-equivalence parser; ambiguous or unsupported forms may fall back, and changed candidates still need human review.",
            "",
            "Context7 confirmed LangChain structured output `include_raw=True` returns raw, parsed, and parsing-error fields, and `json_mode` requires its schema to be described in the prompt: [LangChain structured output docs](https://docs.langchain.com/oss/python/langchain/models). The capture adapter uses the production `QueryNormalizer.from_model` prompt and returns the parsed object to it.",
        ]
        with (OUT / "report.md").open("x", encoding="utf-8") as handle:
            handle.write("\n".join(report) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("initial", "remaining"), required=True)
    run(parser.parse_args().stage)


if __name__ == "__main__":
    main()
