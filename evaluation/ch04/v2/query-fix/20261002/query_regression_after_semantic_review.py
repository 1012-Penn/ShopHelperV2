#!/usr/bin/env python3
"""Four bounded live regressions after semantic review; never rewrites old evidence."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from query_probe import classify, digest_json, guard_record, json_safe, message_dump, read_calibration

DATA = ROOT / "evaluation/ch04/v2"
OUT = DATA / "query-fix/20261002"
IDS = ("V2C003", "V2C006", "V2C009", "V2C012")
RESULTS = OUT / "regression-after-semantic-review.jsonl"
METADATA = OUT / "regression-after-semantic-review.metadata.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_metadata_whitelist(raw):
    metadata = getattr(raw, "response_metadata", {}) or {}
    return {key: json_safe(metadata[key]) for key in
            ("model_name", "finish_reason", "system_fingerprint") if key in metadata}


class CapturedRunnable:
    def __init__(self, runnable, captures):
        self.runnable = runnable
        self.captures = captures

    def invoke(self, messages):
        entry = {"messages": message_dump(messages)}
        try:
            response = self.runnable.invoke(messages)
        except Exception as error:
            entry["request_error_type"] = type(error).__name__
            self.captures.append(entry)
            raise
        raw = response.get("raw")
        parsed = response.get("parsed")
        parsing_error = response.get("parsing_error")
        entry.update({
            "raw_content": json_safe(getattr(raw, "content", None)),
            "response_metadata": response_metadata_whitelist(raw),
            "usage_metadata": json_safe(getattr(raw, "usage_metadata", None)),
            "parsed": parsed.model_dump() if parsed is not None else None,
            "parsing_error_type": type(parsing_error).__name__ if parsing_error else None,
        })
        self.captures.append(entry)
        # Preserve the exact Pydantic value returned by LangChain for production.
        return parsed


class CapturedChatModel:
    def __init__(self, model, captures):
        self.model = model
        self.captures = captures

    def with_structured_output(self, schema, *, method):
        return CapturedRunnable(
            self.model.with_structured_output(schema, method=method, include_raw=True),
            self.captures,
        )


def run():
    if RESULTS.exists() or METADATA.exists():
        raise RuntimeError("regression outputs already exist; refusing duplicate model calls")
    old_run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    _, calibration, by_id, cases_sha = read_calibration()
    cases = [by_id[eval_id] for eval_id in IDS]
    if len(calibration) != 60 or any(case.get("split") != "calibration" for case in cases):
        raise RuntimeError("regressions must use the same frozen calibration cases")

    from app.config import Settings
    from app.services.quality.query import NormalizedQuery, QueryNormalizer
    from langchain_openai import ChatOpenAI

    settings = Settings.from_env()
    model = ChatOpenAI(model=settings.model, api_key=settings.api_key,
                       base_url=settings.base_url, temperature=0, timeout=90,
                       max_retries=0)
    captures = []
    normalizer = QueryNormalizer.from_model(CapturedChatModel(model, captures))
    query_path = ROOT / "app/services/quality/query.py"
    schema = NormalizedQuery.model_json_schema()

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_ids": list(IDS),
        "source_split": "calibration",
        "call_count": 4,
        "test_split_calls": 0,
        "temperature": 0,
        "max_retries": 0,
        "method": "json_mode",
        "include_raw": True,
        "request_model": settings.model,
        "request_endpoint_host": urlsplit(settings.base_url).hostname,
        "old_prompt_sha256": old_run["prompt_sha256"],
        "schema": schema,
        "schema_sha256": digest_json(schema),
        "guard_revision_sha256": sha256_file(query_path),
        "cases_sha256": cases_sha,
        "metadata_policy": "Only model_name, finish_reason, and system_fingerprint are retained from response_metadata; headers are not stored. Parsing errors are recorded by type only.",
    }

    rows = []
    with RESULTS.open("x", encoding="utf-8") as results_handle:
        with METADATA.open("x", encoding="utf-8") as metadata_handle:
            json.dump(metadata, metadata_handle, ensure_ascii=False, indent=2, allow_nan=False)
            metadata_handle.write("\n")
        for case in cases:
            before = len(captures)
            result = normalizer.normalize(case["query"])
            if len(captures) != before + 1:
                raise RuntimeError(f"expected one capture for {case['eval_id']}")
            capture = captures[-1]
            messages = capture.get("messages", [])
            system_prompt = next((message.get("content", "") for message in messages
                                  if message.get("role") == "system"), "")
            candidate = (capture.get("parsed") or {}).get("canonical", "")
            row = {
                "eval_id": case["eval_id"],
                "bucket": case["bucket"],
                "split": case["split"],
                "query": case["query"],
                "candidate": candidate,
                "canonical": result.canonical,
                "downgrade_reason": result.downgrade_reason,
                "outcome": classify(case["query"], candidate, result.canonical, result.downgrade_reason),
                "guard": guard_record(case["query"], candidate),
                "requires_human_semantic_review": result.canonical != case["query"].strip(),
                "messages": messages,
                "raw_content": capture.get("raw_content"),
                "response_metadata": capture.get("response_metadata", {}),
                "provider_response_model": capture.get("response_metadata", {}).get("model_name"),
                "usage_metadata": capture.get("usage_metadata"),
                "parsed": capture.get("parsed"),
                "parsing_error_type": capture.get("parsing_error_type"),
                "request_error_type": capture.get("request_error_type"),
                "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                "schema_sha256": metadata["schema_sha256"],
                "guard_revision_sha256": metadata["guard_revision_sha256"],
                "old_prompt_sha256": metadata["old_prompt_sha256"],
            }
            rows.append(row)
            results_handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            results_handle.flush()

    prompt_hashes = {row["prompt_sha256"] for row in rows}
    if len(prompt_hashes) != 1:
        raise RuntimeError("the production prompt changed during the four regression calls")
    metadata["new_prompt_sha256"] = prompt_hashes.pop()
    if metadata["new_prompt_sha256"] == metadata["old_prompt_sha256"]:
        raise RuntimeError("expected a new prompt hash after the prompt revision")
    metadata["new_system_prompt"] = next(
        message["content"] for message in rows[0]["messages"] if message.get("role") == "system")
    metadata["request_provider_model_pairs"] = [
        {"eval_id": row["eval_id"], "request_model": settings.model,
         "provider_response_model": row["provider_response_model"]}
        for row in rows
    ]

    with METADATA.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    summary = [{"eval_id": row["eval_id"], "outcome": row["outcome"],
                "candidate": row["candidate"], "canonical": row["canonical"],
                "downgrade_reason": row["downgrade_reason"],
                "provider_response_model": row["provider_response_model"]}
               for row in rows]
    print(json.dumps({"metadata": metadata, "results": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
