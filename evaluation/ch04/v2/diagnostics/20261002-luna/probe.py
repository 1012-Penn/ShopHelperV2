#!/usr/bin/env python3
"""Small, read-only probes for the frozen ch04/v2 diagnostics.

Live stages are deliberately split so each can be reviewed before execution:
  offline     inspect frozen rows, SQL hydration and pure guard counterexamples
  normalize   4 DeepSeek structured-output calls
  route       7 embedding calls, 21 Milvus top-50 searches, 2 rerank calls
  generation  4 DeepSeek structured-output calls on historical evidence

No stage calls build_live(), creates tables/collections, indexes, or writes eval.db.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from fractions import Fraction
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "evaluation/ch04/v2"
RUN = DATA / "runs/20261002-120853-23796104"
OUT = DATA / "diagnostics/20261002-luna"
TARGETS = ["V2B021", "V2B036", "V2C022", "V2C034"]
E_TARGETS = ["V2E012", "V2E021", "V2E051", "V2E057", "V2E060", "V2E048", "V2E054"]
NORMALIZE_TARGETS = ["V2A003", "V2B001", "V2C001", "V2C003"]


def digest(value) -> str:
    if isinstance(value, bytes):
        raw = value
    else:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read_inputs():
    cases = json.loads((DATA / "cases.json").read_text())
    case_by_id = {x["eval_id"]: x for x in cases}
    rows = [json.loads(line) for line in (RUN / "rows.jsonl").read_text().splitlines() if line.strip()]
    row_by_pair = {(x["eval_id"], x["strategy"]): x for x in rows}
    manifest = json.loads((DATA / "manifest.json").read_text())
    report = json.loads((RUN / "report.json").read_text())
    return cases, case_by_id, rows, row_by_pair, manifest, report


def verify_frozen():
    from scripts.validate_ch04_dataset import load_corpus, verify_frozen_dataset
    drafts = load_corpus(DATA / "corpus")
    verify_frozen_dataset(DATA, drafts)
    return drafts


def guard_decision(raw: str, candidate: str) -> dict:
    """Mirror QueryNormalizer's current acceptance predicate without calling a model."""
    identifiers = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", raw)
    candidate_identifiers = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate)
    chinese_numbers = re.findall(r"[零〇一二三四五六七八九十百千万亿两]+", raw)
    candidate_numbers = re.findall(r"[零〇一二三四五六七八九十百千万亿两]+", candidate)
    negative = bool(re.search(r"[不无未没非别否]", raw))
    accepted = bool(candidate.strip()) and sorted(identifiers) == sorted(candidate_identifiers) and chinese_numbers == candidate_numbers and (not negative or candidate.strip() == raw.strip())
    failed = []
    if not candidate.strip():
        failed.append("empty_candidate")
    if sorted(identifiers) != sorted(candidate_identifiers):
        failed.append("identifier_mismatch")
    if chinese_numbers != candidate_numbers:
        failed.append("chinese_number_mismatch")
    if negative and candidate.strip() != raw.strip():
        failed.append("negative_query_requires_exact_text")
    return {"accepted": accepted, "failed_predicates": failed,
            "raw_identifiers": identifiers, "candidate_identifiers": candidate_identifiers,
            "raw_chinese_numbers": chinese_numbers, "candidate_chinese_numbers": candidate_numbers,
            "raw_has_negative_character": negative}


def sqlite_readonly():
    path = (DATA / "eval.db").resolve()
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def historical_candidates(con, row, collection):
    ids = row["candidate_ids"]
    placeholders = ",".join("?" for _ in ids)
    chunks = {r[0]: r for r in con.execute(
        f"select id,source_key,questions,answer,category,content_hash,is_active from knowledge_chunks where id in ({placeholders})", ids
    )}
    sync = {r[0]: r[1] for r in con.execute(
        f"select chunk_id,content_hash from knowledge_hybrid_sync where collection=? and chunk_id in ({placeholders})", [collection, *ids]
    )}
    result = []
    for rank, chunk_id in enumerate(ids, 1):
        chunk = chunks.get(chunk_id)
        if chunk is None:
            result.append({"rank": rank, "chunk_id": chunk_id, "hydrated": False, "reason": "missing_sql_row"})
            continue
        valid = bool(chunk[6]) and chunk[5] == sync.get(chunk_id)
        questions = json.loads(chunk[2]) if chunk[2] else []
        result.append({"rank": rank, "chunk_id": chunk_id, "source_key": chunk[1],
                       "question": questions[0] if questions else "", "answer": chunk[3],
                       "category": chunk[4], "content_hash": chunk[5], "hydrated": valid,
                       "reason": None if valid else "inactive_or_hash_mismatch"})
    return result


def offline():
    cases, case_by_id, rows, row_by_pair, manifest, report = read_inputs()
    collection = report["metadata"]["collection"]
    con = sqlite_readonly()
    try:
        synopsis = {}
        downgrade = {}
        for strategy in sorted({x["strategy"] for x in rows}):
            selected = [x for x in rows if x["strategy"] == strategy]
            downgrade[strategy] = {
                "rows": len(selected),
                "with_downgrade_reason": sum(bool(x.get("retrieval_trace", {}).get("downgrade_reason")) for x in selected),
                "canonical_equals_query": sum(x.get("retrieval_trace", {}).get("canonical_query") == x["query"] for x in selected),
            }
        synopsis["normalizer_history"] = downgrade
        synopsis["normalizer_query_features"] = {
            "total": len(cases),
            "has_ascii_identifier": sum(bool(re.search(r"[A-Za-z0-9][A-Za-z0-9_.-]*", x["query"])) for x in cases),
            "has_chinese_number_run": sum(bool(re.search(r"[零〇一二三四五六七八九十百千万亿两]+", x["query"])) for x in cases),
            "has_negative_character": sum(bool(re.search(r"[不无未没非别否]", x["query"])) for x in cases),
        }
        synopsis["synthetic_guard_cases"] = [
            {"name": "identifier_drop", "raw": "MX-405-SE接口是什么", "candidate": "MX-405接口是什么"},
            {"name": "chinese_number_change", "raw": "MX-474三天内能退吗", "candidate": "MX-474五天内能退吗"},
            {"name": "negative_legal_paraphrase", "raw": "MX-474不是快充吧？", "candidate": "MX-474是否不支持快充？"},
            {"name": "false_negative_character_bie", "raw": "MX-474分别是什么接口？", "candidate": "MX-474接口是什么？"},
            {"name": "false_chinese_number_yidian", "raw": "MX-474这个型号能快一点吗？", "candidate": "MX-474这个型号速度能快些吗？"},
            {"name": "repeated_identifier_drop", "raw": "MX-474Pro和MX-474Pro怎么选？", "candidate": "MX-474Pro怎么选？"},
            {"name": "positive_control", "raw": "MX-405的接口是什么", "candidate": "MX-405的接口类型是什么"},
        ]
        for item in synopsis["synthetic_guard_cases"]:
            item["decision"] = guard_decision(item["raw"], item["candidate"])
            from app.services.quality.query import QueryNormalizer
            prod = QueryNormalizer(rewrite=lambda _raw, c=item["candidate"]: c).normalize(item["raw"])
            item["production_normalizer_result"] = {"canonical": prod.canonical, "downgrade_reason": prod.downgrade_reason}
            item["guard_matches_production"] = (prod.canonical == item["candidate"].strip()) == item["decision"]["accepted"]
        synopsis["routes"] = {}
        for eval_id in E_TARGETS + ["V2C001"]:
            row = row_by_pair[(eval_id, "hybrid_rerank") if eval_id != "V2C001" else (eval_id, "bm25")]
            case = case_by_id[eval_id]
            full = historical_candidates(con, row, collection)
            positions = {gold: [x["rank"] for x in full if x.get("source_key") == gold] for gold in case["relevant_source_keys"]}
            synopsis["routes"][eval_id] = {
                "historical_strategy": row["strategy"], "query": case["query"],
                "canonical_query": row.get("retrieval_trace", {}).get("canonical_query", case["query"]),
                "lexical_query": row.get("retrieval_trace", {}).get("lexical_query", case["query"]),
                "category": case.get("category"), "candidate_count": len(row.get("candidate_ids", [])),
                "gold_candidate_ranks": positions,
                "hydrated_candidates": sum(bool(x.get("hydrated")) for x in full),
                "candidate_order": [{k: x.get(k) for k in ("rank", "chunk_id", "source_key", "content_hash", "hydrated")} for x in full],
                "candidates": full,
            }
        synopsis["protocol_history"] = {}
        for eval_id in TARGETS:
            row = row_by_pair[(eval_id, "hybrid_rerank")]
            cite = row.get("citations", [])
            synopsis["protocol_history"][eval_id] = {
                "refusal_reason": row.get("refusal_reason"), "historical_answer": row.get("answer"),
                "citation_count": len(cite), "evidence_sha256": digest(cite),
                "evidence": cite, "query": case_by_id[eval_id]["query"],
                "historical_final_ids": row.get("final_ids"),
            }
        synopsis["input_hashes"] = {
            "cases.json": digest((DATA / "cases.json").read_bytes()),
            "manifest.json": digest((DATA / "manifest.json").read_bytes()),
            "rows.jsonl": digest((RUN / "rows.jsonl").read_bytes()),
            "report.json": digest((RUN / "report.json").read_bytes()),
            "eval_db_file": digest((DATA / "eval.db").read_bytes()),
        }
        synopsis["frozen_manifest_sha256"] = manifest.get("dataset_sha256")
        synopsis["historical_collection"] = collection
        drafts = verify_frozen()
        from app.services.knowledge.content import build_knowledge_fingerprint
        expected = {d.source_key: (d.category, d.questions, d.answer,
                                   build_knowledge_fingerprint(d.category, d.questions, d.answer)) for d in drafts}
        db_matches = {}
        for route in synopsis["routes"].values():
            for item in route["candidates"]:
                if item.get("source_key") not in expected or not item.get("hydrated"):
                    continue
                category, questions, answer, content_hash = expected[item["source_key"]]
                db_matches[item["source_key"]] = db_matches.get(item["source_key"], True) and (
                    item["category"] == category and json.loads(con.execute(
                        "select questions from knowledge_chunks where source_key=?", (item["source_key"],)
                    ).fetchone()[0]) == questions and item["answer"] == answer and item["content_hash"] == content_hash)
        synopsis["frozen_corpus_sql_matches"] = {"checked_sources": len(db_matches), "all_match": all(db_matches.values()),
                                                  "mismatches": sorted(k for k, v in db_matches.items() if not v)}
        synopsis["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    finally:
        con.close()
    save("offline.json", synopsis)
    print(json.dumps({"stage": "offline", "collection": collection, "input_hashes": synopsis["input_hashes"],
                      "normalizer_history": synopsis["normalizer_history"],
                      "normalizer_query_features": synopsis["normalizer_query_features"],
                      "gold_ranks": {k: v["gold_candidate_ranks"] for k, v in synopsis["routes"].items()}}, ensure_ascii=False, indent=2))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def chat_model(settings):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=settings.model, api_key=settings.api_key, base_url=settings.base_url,
                      temperature=0, timeout=90, max_retries=0)


def live_meta(settings):
    return {"model": settings.model, "endpoint_host": urlsplit(settings.base_url).hostname,
            "temperature": 0, "max_retries": 0, "generated_at_utc": datetime.now(timezone.utc).isoformat()}


def normalize_live():
    verify_frozen()
    from app.config import Settings
    from app.services.quality.query import NormalizedQuery, QueryNormalizer
    settings = Settings.from_env()
    structured = chat_model(settings).with_structured_output(NormalizedQuery, method="json_mode", include_raw=True)
    cases, case_by_id, *_ = read_inputs()
    captures = []
    for eval_id in NORMALIZE_TARGETS:
        query = case_by_id[eval_id]["query"]
        response = structured.invoke([
            {"role": "system", "content": '将本轮口语客服问题归一为标准问法，只看本轮。不添加型号、数字、品类，不丢否定条件。输出JSON：{"canonical":"标准问法"}。'},
            {"role": "user", "content": query},
        ])
        parsed = response.get("parsed")
        candidate = parsed.canonical if parsed is not None else ""
        item = {"eval_id": eval_id, "query": query, "raw_content": getattr(response.get("raw"), "content", None),
                "parsed": parsed.model_dump() if parsed is not None else None,
                "parsing_error_type": type(response["parsing_error"]).__name__ if response.get("parsing_error") else None,
                "guard": guard_decision(query, candidate)}
        normalizer = QueryNormalizer(rewrite=lambda _: candidate)
        result = normalizer.normalize(query)
        item["production_normalizer_result"] = {"canonical": result.canonical, "lexical": result.lexical,
                                                "downgrade_reason": result.downgrade_reason}
        captures.append(item)
    save("normalize.json", {"model": live_meta(settings), "inputs_sha256": digest([case_by_id[x] for x in NORMALIZE_TARGETS]),
                             "synthetic_counterexamples": json.loads((OUT / "offline.json").read_text())["synthetic_guard_cases"],
                             "results": captures})


def hydrate_hit_rows(hits, con, collection):
    ids = [h.chunk_id for h in hits]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    chunks = {r[0]: r for r in con.execute(
        f"select id,source_key,questions,answer,category,content_hash,is_active from knowledge_chunks where id in ({placeholders})", ids
    )}
    sync = {r[0]: r[1] for r in con.execute(
        f"select chunk_id,content_hash from knowledge_hybrid_sync where collection=? and chunk_id in ({placeholders})", [collection, *ids]
    )}
    out = []
    for rank, hit in enumerate(hits, 1):
        row = chunks.get(hit.chunk_id)
        if row is None or not row[6] or sync.get(hit.chunk_id) != hit.content_hash or row[5] != hit.content_hash:
            out.append({"rank": rank, "chunk_id": hit.chunk_id, "score": hit.score, "hydrated": False})
            continue
        questions = json.loads(row[2]) if row[2] else []
        out.append({"rank": rank, "chunk_id": hit.chunk_id, "score": hit.score, "source_key": row[1],
                    "question": questions[0] if questions else "", "answer": row[3], "category": row[4],
                    "content_hash": row[5], "hydrated": True})
    return out


def route_live():
    verify_frozen()
    from app.config import Settings
    from openai import OpenAI
    from app.services.knowledge.embeddings import EmbeddingClient
    from app.services.knowledge.hybrid_store import HybridStore
    from app.services.quality.rerank import Reranker
    from dotenv import dotenv_values
    settings = Settings.from_env()
    _, case_by_id, _, row_by_pair, _, report = read_inputs()
    collection = report["metadata"]["collection"]
    values = {k: v for k, v in dotenv_values(".env").items() if v is not None}; values.update(os.environ)
    if collection != "knowledge_ch04_eval_v2":
        raise RuntimeError("frozen run does not point to expected v2 evaluation collection")
    embeds = EmbeddingClient(api_key=settings.embedding_api_key, base_url=settings.embedding_api_base,
                             model=settings.embedding_model, client=OpenAI(api_key=settings.embedding_api_key,
                             base_url=settings.embedding_api_base, max_retries=0))
    store = HybridStore(uri=settings.milvus_uri or "http://localhost:19530", collection_name=collection)
    reranker = Reranker(values.get("RERANK_API_KEY", ""), values.get("RERANK_API_BASE", "https://api.siliconflow.cn/v1"))
    con = sqlite_readonly()
    results = {}
    try:
        for eval_id in E_TARGETS:
            case = case_by_id[eval_id]
            historical = row_by_pair[(eval_id, "hybrid_rerank")]
            trace = historical.get("retrieval_trace", {})
            query = trace.get("canonical_query", case["query"])
            lexical = trace.get("lexical_query", case["query"])
            vector = embeds.embed_query(query)
            runs = {}
            for strategy in ("dense", "bm25", "hybrid"):
                hits = store.search(vector if strategy != "bm25" else None, lexical, strategy,
                                    category=case.get("category"), limit=50)
                runs[strategy] = hydrate_hit_rows(hits, con, collection)
            gold = set(case["relevant_source_keys"])
            results[eval_id] = {"query": case["query"], "canonical_query": query, "lexical_query": lexical,
                                "category": case.get("category"), "embedding_dimensions": len(vector),
                                "historical_candidate_ids": historical["candidate_ids"],
                                "routes": {s: {"ids": [x["chunk_id"] for x in data],
                                               "gold_ranks": {k: [x["rank"] for x in data if x.get("source_key") == k] for k in gold},
                                               "items": data} for s, data in runs.items()}}
            if eval_id in {"V2E048", "V2E054"}:
                historical_items = historical_candidates(con, historical, collection)
                if len(historical_items) != 50 or not all(x.get("hydrated") for x in historical_items):
                    raise RuntimeError(f"{eval_id}: historical candidate50 cannot be faithfully hydrated")
                ranked = reranker.rank(query, [x["question"] + "\n" + x["answer"] for x in historical_items], top_n=50)
                full = []
                for rank, (idx, score) in enumerate(ranked, 1):
                    original = historical_items[idx]
                    full.append({"rank": rank, "historical_candidate_rank": idx + 1, "chunk_id": original["chunk_id"],
                                 "source_key": original["source_key"], "score": score,
                                 "gold": original["source_key"] in gold, "content_hash": original["content_hash"]})
                results[eval_id]["historical_rerank_full50"] = full
                results[eval_id]["historical_rerank_gold_ranks"] = {k: [x["rank"] for x in full if x["source_key"] == k] for k in gold}
    finally:
        con.close(); store.close(); embeds.close(); reranker.close()
    save("route.json", {"model": settings.embedding_model, "reranker": "BAAI/bge-reranker-v2-m3",
                         "endpoint_hosts": {"embedding": urlsplit(settings.embedding_api_base).hostname,
                                            "reranker": urlsplit(values.get("RERANK_API_BASE", "https://api.siliconflow.cn/v1")).hostname},
                         "embedding_max_retries": 0, "rerank_max_retries": 0,
                         "collection": collection, "search_calls": 21, "embedding_calls": 7,
                         "rerank_calls": 2, "input_hashes": json.loads((OUT / "offline.json").read_text())["input_hashes"],
                         "results": results})


def rrf_audit():
    route = json.loads((OUT / "route.json").read_text())
    _, _, rows, row_by_pair, _, _ = read_inputs()
    output = {"rrf_k": 60, "formula": "sum(1 / (k + one_based_rank))", "strict_rank_rule": "1 + count(other_score > candidate_score)",
              "provenance": "official PyMilvus RRFRanker(k=60) default; computed from the already captured dense/BM25 lists; no retrieval request",
              "documentation": "https://github.com/milvus-io/pymilvus/blob/master/_autodocs/api-reference/rankers.md",
              "cases": {}}
    for eval_id, result in route["results"].items():
        dense = {x["chunk_id"]: x["rank"] for x in result["routes"]["dense"]["items"]}
        bm25 = {x["chunk_id"]: x["rank"] for x in result["routes"]["bm25"]["items"]}
        hybrid = {x["chunk_id"]: x for x in result["routes"]["hybrid"]["items"]}
        union = set(dense) | set(bm25)
        scores = {chunk_id: (Fraction(1, 60 + dense[chunk_id]) if chunk_id in dense else Fraction(0, 1)) +
                             (Fraction(1, 60 + bm25[chunk_id]) if chunk_id in bm25 else Fraction(0, 1)) for chunk_id in union}
        ordered = sorted(union, key=lambda chunk_id: (-scores[chunk_id], dense.get(chunk_id, 10**6), bm25.get(chunk_id, 10**6), chunk_id))
        strict_ranks = {chunk_id: 1 + sum(score > scores[chunk_id] for score in scores.values()) for chunk_id in union}
        predicted = set(ordered[:50])
        actual = set(hybrid)
        historical_row = row_by_pair[(eval_id, "hybrid_rerank")]
        historical_candidates = set(historical_row.get("candidate_ids", []))
        historical_candidate_order = historical_row.get("candidate_ids", [])
        historical_final = historical_row.get("final_ids", [])
        current_rerank = result.get("historical_rerank_full50", [])
        current_top10_ids = [x["chunk_id"] for x in current_rerank[:10]]
        gold_keys = sorted({key for strategy in result["routes"].values() for key in strategy["gold_ranks"]})
        gold = {}
        for key in gold_keys:
            ids = [chunk_id for chunk_id in union if next((item.get("source_key") for strategy in result["routes"].values()
                                                          for item in strategy["items"] if item["chunk_id"] == chunk_id), None) == key]
            gold[key] = [{"chunk_id": chunk_id, "dense_rank": dense.get(chunk_id), "bm25_rank": bm25.get(chunk_id),
                          "rrf_score": float(scores[chunk_id]), "strict_gt_rank_plus_one": strict_ranks[chunk_id],
                          "in_hybrid50": chunk_id in actual, "actual_hybrid_rank": hybrid[chunk_id]["rank"] if chunk_id in hybrid else None}
                         for chunk_id in ids]
        strict_order_violations = 0
        actual_list = [x["chunk_id"] for x in result["routes"]["hybrid"]["items"]]
        for i, left in enumerate(actual_list):
            for right in actual_list[i + 1:]:
                if scores[left] < scores[right]:
                    strict_order_violations += 1
        output["cases"][eval_id] = {
            "dense_count": len(dense), "bm25_count": len(bm25), "union_count": len(union), "hybrid_count": len(hybrid),
            "hybrid_overlap_with_formula_top50": len(actual & predicted), "hybrid_only_vs_formula": sorted(actual - predicted),
            "formula_only_vs_hybrid": sorted(predicted - actual),
            "hybrid_score_matches_formula": all(abs(hybrid[x]["score"] - float(scores[x])) < 1e-7 for x in actual),
            "historical_candidate_overlap_current_hybrid": len(historical_candidates & actual),
            "historical_candidate_order_equals_current_hybrid": historical_candidate_order == [x["chunk_id"] for x in result["routes"]["hybrid"]["items"]],
            "historical_candidate_moved_positions": [{"chunk_id": chunk_id, "historical_rank": old_rank,
                                                       "current_rank": next((i + 1 for i, x in enumerate(result["routes"]["hybrid"]["items"])
                                                                             if x["chunk_id"] == chunk_id), None)}
                                                      for old_rank, chunk_id in enumerate(historical_candidate_order, 1)
                                                      if next((i + 1 for i, x in enumerate(result["routes"]["hybrid"]["items"])
                                                               if x["chunk_id"] == chunk_id), None) != old_rank],
            "historical_candidate_only_vs_current": sorted(historical_candidates - actual),
            "current_candidate_only_vs_historical": sorted(actual - historical_candidates),
            "historical_final_ids": historical_final,
            "current_full_rerank_top10_ids": current_top10_ids,
            "current_top10_equals_historical_final_ids": current_top10_ids == historical_final if current_rerank else None,
            "strict_order_violations_among_hybrid": strict_order_violations,
            "gold_candidates": gold,
            "candidate_cutoff": [{"chunk_id": x, "rrf_score": float(scores[x]), "strict_gt_rank_plus_one": strict_ranks[x]}
                                 for x in ordered if strict_ranks[x] > 45][:10],
        }
    output["route_input_sha256"] = digest((OUT / "route.json").read_bytes())
    save("rrf-audit.json", output)
    print(json.dumps({k: v for k, v in output.items() if k != "cases"}, ensure_ascii=False, indent=2))
    for eval_id, result in output["cases"].items():
        print(eval_id, "overlap", result["hybrid_overlap_with_formula_top50"], "/", min(50, result["union_count"]),
              "score_matches", result["hybrid_score_matches_formula"],
              "order_violations", result["strict_order_violations_among_hybrid"], "gold", result["gold_candidates"])


def protocol_live(iteration):
    from app.config import Settings
    from app.services.quality.generation import GenerationResult, QUALITY_PROMPT, KnowledgeAnswerService, arrange_evidence
    from app.services.quality.retrieval import Evidence
    settings = Settings.from_env()
    verify_frozen()
    structured = chat_model(settings).with_structured_output(GenerationResult, method="json_mode", include_raw=True)
    _, case_by_id, _, row_by_pair, _, _ = read_inputs()
    results = {}
    for eval_id in TARGETS:
        case = case_by_id[eval_id]
        row = row_by_pair[(eval_id, "hybrid_rerank")]
        evidence = [Evidence(n=x["n"], chunk_id=x["chunk_id"], section_path=x["section_path"],
                             question=x["question"], answer=x["answer"], source_key=x["source_key"],
                             score=x["score"], category=x.get("category", "")) for x in row["citations"]]
        arranged = arrange_evidence(evidence)
        payload = {"question": case["query"], "evidence": [x.snapshot() for x in arranged]}
        response = structured.invoke([{"role": "system", "content": QUALITY_PROMPT},
                                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        parsed = response.get("parsed")
        checked = parsed if isinstance(parsed, GenerationResult) else None
        failure_predicates = []
        parse_error = response.get("parsing_error")
        if checked is None and parse_error is not None:
            failure_predicates.append("structure_parse_failure")
        elif checked is None:
            failure_predicates.append("parsed_result_missing")
        elif not checked.sufficient:
            failure_predicates.append("sufficient_false")
        else:
            numbers = set(map(int, re.findall(r"\[(\d+)\]", checked.answer)))
            allowed = {e.n for e in evidence}
            if not checked.answer.strip(): failure_predicates.append("empty_answer")
            if not numbers: failure_predicates.append("no_inline_citation_numbers")
            if not numbers <= allowed: failure_predicates.append("inline_citation_out_of_bounds")
            if numbers != set(checked.cited_numbers): failure_predicates.append("inline_vs_declared_citations_mismatch")
        def replay_generator(_question, _evidence):
            if parse_error is not None:
                raise parse_error
            return parsed
        service_result = KnowledgeAnswerService(None, replay_generator, None, min_score=0.0).generate(
            case["query"], evidence, strategy="hybrid")
        raw = response.get("raw")
        raw_content = getattr(raw, "content", None)
        if isinstance(raw_content, list):
            raw_content = [{k: x.get(k) for k in ("type", "text")} if isinstance(x, dict) else str(x) for x in raw_content]
        results[eval_id] = {"query": case["query"], "historical_refusal_reason": row.get("refusal_reason"),
                            "historical_answer": row.get("answer"),
                            "historical_evidence_sha256": digest(row["citations"]),
                            "historical_evidence": row["citations"],
                            "generation_payload_sha256": digest(payload), "generation_payload": payload,
                            "raw_content": raw_content,
                            "parsed": checked.model_dump() if checked is not None else None,
                            "parsing_error_type": type(parse_error).__name__ if parse_error else None,
                            "replay_failure_predicates": failure_predicates,
                            "production_guard_result": {"refused": service_result.refused, "reason": service_result.reason,
                                                         "answer": service_result.answer}}
    filename = "generation.json" if iteration == 1 else f"generation-repeat{iteration}.json"
    save(filename, {"iteration": iteration, "model": live_meta(settings), "method": "json_mode", "include_raw": True,
                              "max_retries": 0, "prompt_sha256": digest(QUALITY_PROMPT),
                              "results": results})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("offline", "normalize", "route", "rrf-audit", "generation"), required=True)
    parser.add_argument("--iteration", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    if args.stage == "offline": offline()
    elif args.stage == "normalize": normalize_live()
    elif args.stage == "route": route_live()
    elif args.stage == "rrf-audit": rrf_audit()
    elif args.stage == "generation": protocol_live(args.iteration)


if __name__ == "__main__":
    main()
