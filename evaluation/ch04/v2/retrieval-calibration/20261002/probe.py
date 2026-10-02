#!/usr/bin/env python3
"""Calibration-only experiment for the hybrid fusion output limit.

The live stage compares fusion limits 50 and 100. Each underlying dense and
BM25 AnnSearchRequest remains fixed at 50, the RRF ranker and fixed reranker
remain unchanged, and only frozen calibration cases are accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "evaluation/ch04/v2"
QUERY_FIX = DATA / "query-fix/20261002"
DEFAULT_QUERY_INPUT = QUERY_FIX / "regraded-calibration.jsonl"
RUN = DATA / "runs/20261002-120853-23796104/report.json"
OUTPUT = DATA / "retrieval-calibration/20261002"
RESULTS = OUTPUT / "results.json"
LIMITS = (50, 100)
FINAL_TOP_N = 10
MAX_EMBEDDING_CALLS = 60
MAX_HYBRID_SEARCH_CALLS = 120
MAX_RERANK_CALLS = 120
COLLECTION = "knowledge_ch04_eval_v2"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def select_calibration_cases(cases: list[dict]) -> list[dict]:
    if len(cases) != 300:
        raise ValueError(f"expected frozen 300-case dataset, got {len(cases)}")
    ids = [case.get("eval_id") for case in cases]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("case identifiers are missing or duplicated")
    selected = [case for case in cases if case.get("split") == "calibration"]
    if len(selected) != 60 or any(case.get("split") not in {"calibration", "test"} for case in cases):
        raise ValueError(f"expected exactly 60 calibration cases, got {len(selected)}")
    return selected


def metrics_for_sources(final_source_keys, candidate_source_keys, gold_source_keys,
                        should_refuse: bool, candidate_limit: int) -> dict:
    """Use the project retrieval metric implementation; unknown questions stay N/A."""
    from app.services.quality.evaluation import retrieval_metrics

    gold = [] if should_refuse else list(gold_source_keys)
    return {
        "final": retrieval_metrics(list(final_source_keys), gold, ks=(1, 5, 10)),
        "candidate": retrieval_metrics(list(candidate_source_keys), gold, ks=(candidate_limit,)),
    }


def parse_query_inputs(cases: list[dict], rows: list[dict]) -> dict[str, dict]:
    """Validate supplied canonical and lexical strings; never rewrite locally."""
    calibration = select_calibration_cases(cases)
    calibration_by_id = {case["eval_id"]: case for case in calibration}
    if len(rows) != 60 or {row.get("eval_id") for row in rows} != set(calibration_by_id):
        raise ValueError("query input must contain each frozen calibration case exactly once")
    captures = {}
    for row in rows:
        eval_id = row["eval_id"]
        case = calibration_by_id[eval_id]
        if row.get("split", "calibration") != "calibration" or row.get("query") != case["query"]:
            raise ValueError(f"query input does not match frozen calibration case: {eval_id}")
        canonical = row.get("canonical_query", row.get("canonical"))
        lexical = row.get("lexical_query", row.get("lexical"))
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(f"final canonical query is missing: {eval_id}")
        if not isinstance(lexical, str) or not lexical.strip():
            raise ValueError(f"final lexical query is missing: {eval_id}")
        if canonical != canonical.strip() or lexical != lexical.strip():
            raise ValueError(f"query input contains untrimmed canonical or lexical text: {eval_id}")
        provenance = {key: value for key, value in row.items() if key not in {
            "eval_id", "split", "query", "canonical", "canonical_query", "lexical", "lexical_query"}}
        captures[eval_id] = {
            "canonical_query": canonical,
            "lexical_query": lexical,
            "provenance": provenance,
        }
    return captures


def safe_error_record(error: Exception) -> dict:
    """Keep provider exception text (which can contain request details) out of artifacts."""
    return {"type": type(error).__name__,
            "message": "details omitted to protect provider configuration"}


def load_query_inputs(cases: list[dict], path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Load the parent-reviewed JSONL and pin its exact bytes into the run."""
    cases_path = DATA / "cases.json"
    if not path.is_file():
        raise FileNotFoundError(f"reviewed query input is not ready: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    captures = parse_query_inputs(cases, rows)
    hashes = {
        "cases.json": sha256_file(cases_path),
        "query_input_jsonl": sha256_file(path),
        "query_input_path": str(path.resolve().relative_to(ROOT)),
    }
    hashes["query_input_bundle"] = digest_json({key: hashes[key] for key in sorted(hashes)})
    return captures, hashes


def readonly_db(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def source_id_map(connection: sqlite3.Connection) -> dict[str, list[int]]:
    values = {}
    for chunk_id, source_key in connection.execute(
            "select id, source_key from knowledge_chunks where is_active=1 order by id"):
        values.setdefault(source_key, []).append(int(chunk_id))
    return values


def verify_frozen_corpus() -> int:
    """Check manifest case and corpus hashes through the repository validator."""
    from scripts.validate_ch04_dataset import load_corpus, verify_frozen_dataset

    drafts = load_corpus(DATA / "corpus")
    verify_frozen_dataset(DATA, drafts)
    return len(drafts)


def hydrate_hits(connection: sqlite3.Connection, hits, collection: str) -> tuple[list[dict], list[dict]]:
    """Mirror QualityRetriever.hydrate against read-only SQLite rows."""
    if not hits:
        return [], []
    ids = list(dict.fromkeys(hit.chunk_id for hit in hits))
    marks = ",".join("?" for _ in ids)
    rows = {
        int(row[0]): row for row in connection.execute(
            f"select id,source_key,questions,answer,category,chapter_path,content_hash,is_active "
            f"from knowledge_chunks where id in ({marks}) and is_active=1", ids)
    }
    sync = {
        int(row[0]): row[1] for row in connection.execute(
            f"select chunk_id,content_hash from knowledge_hybrid_sync "
            f"where collection=? and chunk_id in ({marks})", [collection, *ids])
    }
    accepted, dropped = [], []
    for rank, hit in enumerate(hits, 1):
        row = rows.get(hit.chunk_id)
        reason = None
        if row is None:
            reason = "missing_or_inactive_sql_row"
        elif row[6] != hit.content_hash or sync.get(hit.chunk_id) != row[6]:
            reason = "content_hash_or_sync_mismatch"
        if reason:
            dropped.append({"rank": rank, "chunk_id": hit.chunk_id, "content_hash": hit.content_hash,
                            "reason": reason})
            continue
        questions = json.loads(row[2]) if row[2] else []
        section_path = json.loads(row[5]) if row[5] else []
        accepted.append({
            "rank": len(accepted) + 1,
            "hybrid_hit_rank": rank,
            "chunk_id": int(row[0]),
            "source_key": row[1],
            "question": questions[0] if questions else "",
            "answer": row[3],
            "category": row[4],
            "section_path": section_path,
            "content_hash": row[6],
            "rrf_score": float(hit.score),
        })
    return accepted, dropped


def candidate_documents(candidates: list[dict]) -> list[str]:
    return [item["question"] + "\n" + item["answer"] for item in candidates]


def final_snapshot(candidates: list[dict], ranked: list[tuple[int, float]]) -> list[dict]:
    result = []
    for rank, (index, score) in enumerate(ranked, 1):
        item = dict(candidates[index])
        item["rank"] = rank
        item["rerank_score"] = float(score)
        result.append(item)
    return result


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def summarize_variant(rows: list[dict], limit: int) -> dict:
    known = [row for row in rows if not row["should_refuse"] and row["gold_source_keys"]]
    final_metric = lambda row, name: row["variants"][str(limit)]["metrics"]["final"].get(name)
    candidate_metric = lambda row, name: row["variants"][str(limit)]["metrics"]["candidate"].get(name)
    timers = [row["variants"][str(limit)]["timing_seconds"] for row in rows]
    return {
        "cases": len(rows),
        "known_gold_cases": len(known),
        "unknown_cases_metrics_na": sum(row["should_refuse"] for row in rows),
        "final": {name: _mean([final_metric(row, name) for row in known])
                  for name in ("recall@1", "recall@5", "recall@10", "all_evidence@10", "mrr")},
        "candidate_gold_coverage": {
            "recall": _mean([candidate_metric(row, f"recall@{limit}") for row in known]),
            "all_evidence": _mean([candidate_metric(row, f"all_evidence@{limit}") for row in known]),
        },
        "mean_candidate_count": _mean([row["variants"][str(limit)]["candidate_count"] for row in rows]),
        "mean_timing_seconds": {
            name: _mean([timing.get(name) for timing in timers])
            for name in ("embedding", "recall", "hydration", "rerank")
        },
    }


def paired_summary(rows: list[dict], left: int, right: int) -> dict:
    output = {}
    for scope, metric_names in (
        ("final", ("recall@10", "all_evidence@10", "mrr")),
        ("candidate_gold_coverage", ("recall", "all_evidence")),
    ):
        output[scope] = {}
        for metric_name in metric_names:
            deltas = []
            for row in rows:
                if row["should_refuse"] or not row["gold_source_keys"]:
                    continue
                a = row["variants"][str(left)]["metrics"]
                b = row["variants"][str(right)]["metrics"]
                if scope == "final":
                    av, bv = a["final"].get(metric_name), b["final"].get(metric_name)
                else:
                    av = a["candidate"].get(f"{metric_name}@{left}")
                    bv = b["candidate"].get(f"{metric_name}@{right}")
                if av is not None and bv is not None:
                    deltas.append(bv - av)
            output[scope][metric_name] = {
                "pairs": len(deltas),
                "50_wins": sum(value < -1e-12 for value in deltas),
                "ties": sum(abs(value) <= 1e-12 for value in deltas),
                "100_wins": sum(value > 1e-12 for value in deltas),
                "mean_delta_100_minus_50": _mean(deltas),
            }
    return output


def summarize(rows: list[dict]) -> dict:
    return {"candidate_limit_50": summarize_variant(rows, 50),
            "candidate_limit_100": summarize_variant(rows, 100),
            "paired_100_minus_50": paired_summary(rows, 50, 100)}


def make_report(results: dict) -> str:
    summary = results.get("summary", {})
    metadata = results.get("metadata", {})
    lines = [
        "# ch04 hybrid fusion output-limit calibration",
        "",
        f"状态：`{results.get('status')}`；split：calibration 60；test：0。",
        "",
        "本实验只改变融合输出 `limit`（50 / 100）。Milvus dense 与 native BM25 每路仍 limit=50，RRF、`k=60`、最终重排 top_n=10、reranker 与 0.05 阈值均不变。未知题的 gold 检索指标为 N/A。延迟只计 embedding、两次 hybrid recall、hydration 和 rerank，不含改写、生成、裁判或 normalization。",
        "",
        "| 融合输出上限 | 有效 gold 题 | Recall@10 | 完整证据@10 | MRR | 候选 gold Recall | 候选完整 gold | 平均候选数 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for limit in (50, 100):
        item = summary.get(f"candidate_limit_{limit}", {})
        final = item.get("final", {})
        coverage = item.get("candidate_gold_coverage", {})
        fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
        lines.append(
            f"| {limit} | {item.get('known_gold_cases', 0)} | {fmt(final.get('recall@10'))} | "
            f"{fmt(final.get('all_evidence@10'))} | {fmt(final.get('mrr'))} | "
            f"{fmt(coverage.get('recall'))} | {fmt(coverage.get('all_evidence'))} | "
            f"{fmt(item.get('mean_candidate_count'))} |"
        )
    lines += ["", "## 平均阶段耗时（秒）", "", "| 上限 | embedding | recall | hydration | rerank |",
              "|---:|---:|---:|---:|---:|"]
    for limit in (50, 100):
        times = summary.get(f"candidate_limit_{limit}", {}).get("mean_timing_seconds", {})
        fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
        lines.append(f"| {limit} | {fmt(times.get('embedding'))} | {fmt(times.get('recall'))} | "
                     f"{fmt(times.get('hydration'))} | {fmt(times.get('rerank'))} |")
    lines += ["", "## 请求上限", "",
              "最大 embedding 调用 60、Milvus hybrid_search 调用 120、rerank 调用 120；每题一个向量供两个融合查询复用，两个融合结果各至多重排一次。没有改写、生成或裁判请求；异常即停止，脚本不重试。SQLite 使用只读 URI；不创建表、索引、集合或账本。",
              "", "输入及逐题完整候选、重排结果、gold ID/source key、分段耗时和 hashes 见 `results.json`。",
              "", "## API 与实现依据", "",
              "PyMilvus 文档分别定义了每个 `AnnSearchRequest.limit` 和融合 `hybrid_search(limit=...)`，并记录 `RRFRanker(k=60)` 默认值：[AnnSearchRequest / RRFRanker](https://github.com/milvus-io/pymilvus/blob/master/_autodocs/api-reference/rankers.md)、[hybrid_search](https://github.com/milvus-io/pymilvus/blob/master/_autodocs/api-reference/async-milvus-client.md)。SiliconFlow 定义 `/rerank` 的 `top_n`：[Create Rerank](https://docs.siliconflow.com/en/api-reference/rerank/create-rerank)。OpenAI SDK 的 `max_retries=0` 关闭自动重试：[OpenAI Python SDK](https://github.com/openai/openai-python/blob/main/README.md)。",
              "", "运行时实现文件 SHA-256："]
    for name, value in metadata.get("implementation_hashes", {}).items():
        lines.append(f"- `{name}`: `{value}`")
    return "\n".join(lines) + "\n"


def offline(query_input_path: Path) -> dict:
    cases_path = DATA / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    corpus_chunks = verify_frozen_corpus()
    calibration = select_calibration_cases(cases)
    db_path = DATA / "eval.db"
    connection = readonly_db(db_path)
    try:
        chunk_count = connection.execute("select count(*) from knowledge_chunks").fetchone()[0]
        sync_count = connection.execute(
            "select count(*) from knowledge_hybrid_sync where collection=?", (COLLECTION,)).fetchone()[0]
        ids_by_source = source_id_map(connection)
        missing_gold = sorted({source for case in calibration if not case["should_refuse"]
                               for source in case["relevant_source_keys"] if source not in ids_by_source})
    finally:
        connection.close()
    if chunk_count != 480 or sync_count != 480 or missing_gold:
        raise ValueError(f"frozen read-only corpus check failed: chunks={chunk_count}, sync={sync_count}, missing_gold={missing_gold}")
    query_ready = query_input_path.is_file()
    query_input_info = {"ready": False}
    if query_ready:
        captures, hashes = load_query_inputs(cases, query_input_path)
        query_input_info = {"ready": True, "cases": len(captures), "hashes": hashes}
    return {
        "status": "ready_for_parent_review" if query_ready else "waiting_for_regraded_query_input",
        "cases_total": len(cases),
        "calibration_cases": len(calibration),
        "test_cases_excluded": sum(case["split"] == "test" for case in cases),
        "unknown_calibration_cases_metrics_na": sum(bool(case["should_refuse"]) for case in calibration),
        "database_chunks": chunk_count,
        "frozen_corpus_chunks": corpus_chunks,
        "hybrid_sync_rows": sync_count,
        "collection": COLLECTION,
        "candidate_limits": list(LIMITS),
        "ann_search_request_limit_each": 50,
        "rrf_k": 60,
        "reranker": RERANK_MODEL,
        "final_top_n": FINAL_TOP_N,
        "min_score_threshold_unchanged": 0.05,
        "max_calls": {"embedding": MAX_EMBEDDING_CALLS,
                      "milvus_hybrid_search": MAX_HYBRID_SEARCH_CALLS,
                      "rerank": MAX_RERANK_CALLS,
                      "query_rewrite_generation_judge": 0},
        "read_only_check": True,
        "reviewed_query_input": query_input_info,
        "cases_sha256": sha256_file(cases_path),
        "eval_db_sha256": sha256_file(db_path),
    }


def read_settings():
    from dotenv import dotenv_values
    values = {key: value for key, value in dotenv_values(ROOT / ".env").items() if value is not None}
    values.update(os.environ)
    required = ("EMBEDDING_API_KEY", "RERANK_API_KEY", "MILVUS_URI")
    if any(not values.get(name, "").strip() for name in required):
        raise ValueError("EMBEDDING_API_KEY, RERANK_API_KEY, and MILVUS_URI are required for the approved live stage")
    return values


def write_results(results: dict) -> None:
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")


def live(query_input_path: Path) -> dict:
    if RESULTS.exists():
        raise FileExistsError("results.json already exists; refusing duplicate live requests")
    cases = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))
    corpus_chunks = verify_frozen_corpus()
    calibration = select_calibration_cases(cases)
    query_inputs, query_hashes = load_query_inputs(cases, query_input_path)
    connection = readonly_db(DATA / "eval.db")
    case_by_id = {case["eval_id"]: case for case in calibration}
    ids_by_source = source_id_map(connection)
    report_metadata = json.loads(RUN.read_text(encoding="utf-8"))
    if report_metadata.get("metadata", {}).get("collection") != COLLECTION:
        connection.close()
        raise ValueError("historical report collection does not match the fixed evaluation collection")

    settings_values = read_settings()
    from openai import OpenAI
    from app.services.knowledge.embeddings import EmbeddingClient
    from app.services.knowledge.hybrid_store import HybridStore
    from app.services.quality.rerank import Reranker

    embedding_model = settings_values.get("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
    embedding_base = settings_values.get("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1").strip().rstrip("/")
    rerank_base = settings_values.get("RERANK_API_BASE", "https://api.siliconflow.cn/v1").strip().rstrip("/")
    embedding_sdk = OpenAI(api_key=settings_values["EMBEDDING_API_KEY"].strip(), base_url=embedding_base,
                           timeout=45, max_retries=0)
    embeddings = EmbeddingClient(api_key=settings_values["EMBEDDING_API_KEY"].strip(),
                                 base_url=embedding_base, model=embedding_model, client=embedding_sdk)
    store = HybridStore(uri=settings_values["MILVUS_URI"].strip(), collection_name=COLLECTION)
    reranker = Reranker(settings_values.get("RERANK_API_KEY", "").strip(), rerank_base)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "calibration-only fusion-output-limit comparison",
        "split": "calibration",
        "calibration_cases": len(calibration),
        "test_cases_requested": 0,
        "limits": list(LIMITS),
        "ann_search_request_limit_each": 50,
        "rrf_k": 60,
        "dense_model": embedding_model,
        "reranker_model": RERANK_MODEL,
        "final_top_n": FINAL_TOP_N,
        "min_score_threshold_unchanged": 0.05,
        "embedding_max_retries": 0,
        "rerank_httpx_retries": 0,
        "embedding_endpoint_host": urlsplit(embedding_base).hostname,
        "rerank_endpoint_host": urlsplit(rerank_base).hostname,
        "collection": COLLECTION,
        "frozen_corpus_chunks": corpus_chunks,
        "call_budget": {"embedding_max": MAX_EMBEDDING_CALLS,
                        "hybrid_search_max": MAX_HYBRID_SEARCH_CALLS,
                        "rerank_max": MAX_RERANK_CALLS,
                        "normalization_generation_judge": 0},
        "input_hashes": {
            **query_hashes,
            "eval.db": sha256_file(DATA / "eval.db"),
            "historical_run_report.json": sha256_file(RUN),
            "script.py": sha256_file(Path(__file__)),
        },
        "implementation_hashes": {
            name: sha256_file(ROOT / name) for name in (
                "app/services/knowledge/hybrid_store.py",
                "app/services/quality/retrieval.py",
                "app/services/quality/rerank.py",
                "app/services/quality/evaluation.py",
            )
        },
        "query_input_bundle_sha256": query_hashes["query_input_bundle"],
        "query_input_path": query_hashes["query_input_path"],
        "results_hash_note": "Each case input hash covers its frozen case, parent-supplied canonical/lexical input and provenance, and the complete reviewed JSONL hash.",
    }
    results = {"status": "running", "metadata": metadata, "call_counts": {
        "embedding": 0, "hybrid_search": 0, "rerank": 0}, "cases": []}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Create a guard artifact before the first request; an interrupted run cannot
    # be accidentally repeated and duplicate the external calls.
    write_results(results)
    error = None
    try:
        for index, case in enumerate(calibration):
            query = query_inputs[case["eval_id"]]
            per_case_hash = digest_json({
                "case": case,
                "canonical_query": query["canonical_query"],
                "lexical_query": query["lexical_query"],
                "query_input_bundle_sha256": query_hashes["query_input_bundle"],
                "query_provenance": query["provenance"],
            })
            embedding_started = time.monotonic()
            results["call_counts"]["embedding"] += 1
            vector = embeddings.embed_query(query["canonical_query"])
            embedding_seconds = time.monotonic() - embedding_started

            limit_order = (LIMITS if index % 2 == 0 else tuple(reversed(LIMITS)))
            variants = {}
            for candidate_limit in limit_order:
                recall_started = time.monotonic()
                results["call_counts"]["hybrid_search"] += 1
                hits = store.search(vector, query["lexical_query"], "hybrid",
                                    category=case.get("category"), limit=candidate_limit)
                recall_seconds = time.monotonic() - recall_started
                hydration_started = time.monotonic()
                candidates, dropped = hydrate_hits(connection, hits, COLLECTION)
                hydration_seconds = time.monotonic() - hydration_started
                rerank_seconds = 0.0
                ranked = []
                if candidates:
                    rerank_started = time.monotonic()
                    results["call_counts"]["rerank"] += 1
                    ranked = reranker.rank(query["canonical_query"], candidate_documents(candidates), FINAL_TOP_N)
                    rerank_seconds = time.monotonic() - rerank_started
                final = final_snapshot(candidates, ranked)
                gold_keys = list(case.get("relevant_source_keys", []))
                metrics = metrics_for_sources(
                    [item["source_key"] for item in final],
                    [item["source_key"] for item in candidates],
                    gold_keys, bool(case["should_refuse"]), candidate_limit)
                gold = [{"source_key": key, "chunk_ids": ids_by_source.get(key, [])}
                        for key in gold_keys]
                variants[str(candidate_limit)] = {
                    "candidate_limit": candidate_limit,
                    "candidate_count": len(candidates),
                    "raw_hit_count": len(hits),
                    "raw_hits": [{"rank": rank, "chunk_id": int(hit.chunk_id),
                                  "rrf_score": float(hit.score), "content_hash": hit.content_hash}
                                 for rank, hit in enumerate(hits, 1)],
                    "dropped_hits": dropped,
                    "candidates": candidates,
                    "final": final,
                    "final_ids": [item["chunk_id"] for item in final],
                    "metrics": metrics,
                    "timing_seconds": {"embedding": embedding_seconds,
                                        "recall": recall_seconds,
                                        "hydration": hydration_seconds,
                                        "rerank": rerank_seconds},
                    "rerank_documents_sent": len(candidates),
                }
            results["cases"].append({
                **{key: case[key] for key in case if key in {
                    "eval_id", "bucket", "split", "difficulty", "family_id", "query", "category",
                    "challenge_tags", "should_refuse"}},
                "canonical_query": query["canonical_query"],
                "lexical_query": query["lexical_query"],
                "query_provenance": query["provenance"],
                "gold_source_keys": gold_keys,
                "gold_ids": gold,
                "input_hash": per_case_hash,
                "input_hash_sha256": per_case_hash,
                "embedding_seconds": embedding_seconds,
                "variants": variants,
            })
            write_results(results)
            print(json.dumps({"eval_id": case["eval_id"], "completed": index + 1,
                              "embedding_calls": results["call_counts"]["embedding"],
                              "hybrid_search_calls": results["call_counts"]["hybrid_search"],
                              "rerank_calls": results["call_counts"]["rerank"]}, ensure_ascii=False), flush=True)
        results["status"] = "complete"
        results["summary"] = summarize(results["cases"])
    except Exception as exc:
        error = safe_error_record(exc)
        results["status"] = "failed"
        results["error"] = error
    finally:
        reranker.close()
        store.close()
        embeddings.close()
        connection.close()
        write_results(results)
    if results["status"] == "complete":
        (OUTPUT / "report.md").write_text(make_report(results), encoding="utf-8")
    if error:
        raise RuntimeError(f"calibration stopped after {results['call_counts']}: {error['type']}") from None
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("offline", "live"), required=True,
                        help="offline performs no network/API calls; live requires prior parent approval")
    parser.add_argument("--query-input", type=Path, default=DEFAULT_QUERY_INPUT,
                        help="reviewed calibration JSONL with one canonical and lexical query per case")
    args = parser.parse_args()
    query_input_path = args.query_input if args.query_input.is_absolute() else ROOT / args.query_input
    if args.stage == "offline":
        print(json.dumps(offline(query_input_path), ensure_ascii=False, indent=2))
    else:
        result = live(query_input_path)
        print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                          "call_counts": result["call_counts"], "summary": result.get("summary")},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
