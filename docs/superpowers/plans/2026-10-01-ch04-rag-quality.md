# ch04 RAG Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver traceable hybrid RAG, evidence refusal, durable case ledgers and a live four-strategy benchmark with at least 300 labeled queries.

**Architecture:** Extend the existing knowledge tool and ChatService with independent query, retrieval and guarded generation services. MySQL remains authoritative; a new Milvus collection records synchronized content hashes. Evaluation uses the same runtime pipeline and persists canonical hybrid_rerank fabrication cases.

**Tech Stack:** MySQL 8, SQLAlchemy 2, FastAPI, LangChain, Milvus 3/PyMilvus 3, BGE-M3 embeddings, SiliconFlow BAAI/bge-reranker-v2-m3, React/Vite.

**Spec:** `docs/superpowers/specs/2026-09-30-ch04-rag-quality-design.md` (approved 2026-10-01).

## Global Constraints

- Recommend native inline execution after plan review; user approved design and said “评估集至少300条. 可以开始工作”.
- Milvus native BM25 Function with builtin chinese analyzer; dense and BM25 each Top-50, hybrid_search RRFRanker, rerank Top-10.
- Keep string conversation_id primary key; add unique unsigned numeric id; supplied low-confidence DDL targets numeric id.
- At least 300 distinct labeled queries: 60 per bucket A_policy/B_model/C_colloquial/D_unknown/E_multi; 12 calibration and 48 test per bucket, with all three difficulties.
- Preserve models/numbers/negation during single-turn rewrite. No anaphora or multi-turn rewriting. Synonyms only at retrieval.
- Frontend uses Vibe Coding: no brainstorming/TDD/reviewer gate; validate build and interaction.
- Query Context7 before using each library API. Fixed choices cannot be replaced without asking the user.
- Append dev-notes/ch04.md after each stage/task/review/finish; prompt/data changes validated on labeled samples.
- Never claim fixture metrics are live quality measurements; no credentials in repository or logs.

## Review Focus

1. Concurrent/repeated migrations and new conversations must produce stable numeric IDs, never max(id)+1 races (Task 1).
2. Interrupted collection rebuild and changed/deleted SQL chunks must not return stale evidence (Task 2).
3. Query normalization must not drop negation or invent model/category; invalid rerank index must fail safely (Task 3).
4. Tool-free policy answers, invalid citations, insufficient evidence and pool insert failures must not leak a successful unsupported answer (Task 4).
5. Recurrent cases, judge errors and refusal-only strategies must not inflate metrics or silently erase earlier dispositions (Task 6).

## Task 1: Numeric conversation IDs, migrations and durable ledgers

**Files:** Modify `app/db/models.py`, `app/db/session.py`; create `app/services/quality/ledger.py`, `scripts/migrate_ch04.py`, `tests/test_quality_ledgers.py`.

**Interfaces:** `migrate_ch04(engine) -> None`; `QualityLedger(session_factory).add_low_confidence(conversation_key: str, raw_question: str, source: str, reason: str) -> int`; `.record_faith_case(case: dict) -> int`; `.resolve(eval_id: str, status: str, resolution: str) -> None`.

- [x] Query Context7 SQLAlchemy MySQL server-generated fields, inspection, upsert and transaction APIs. Use explicit MySQL ALTER to add unique AUTO_INCREMENT id, mark ORM FetchedValue; for SQLite use a documented trigger/sequence adaptation. Existing MySQL database requires migration before create_all.
- [x] Write failing persistence tests with actual SQLite transactions: two string-key conversations get distinct numeric IDs; low-confidence FK resolves the requested conversation; invalid source fails; assistant citations round-trip. Hand-checked case example: first record A01, resolve it, record again => seen_count=2, status=未解决, resolution=None, resolved_at preserved.
```python
def test_recurrence_reopens_case(ledger, case):
    ledger.record_faith_case(case)
    ledger.resolve('A01', '已解决', '补充证据')
    ledger.record_faith_case({**case, 'answer': '第二轮答案'})
    saved = ledger.list_cases()[0]
    assert (saved.seen_count, saved.status, saved.resolution) == (2, '未解决', None)
    assert saved.resolved_at is not None
```
- [x] Run `python3 -m pytest tests/test_quality_ledgers.py -q` and observe absent implementation failure.
- [x] Implement supplied table fields/enums/indexes, Message.citations, explicit idempotent migration, transactional upsert and disposition validation. Expose `list_cases(status=None)` for CLI. MySQL generated IDs refreshed by query, no unsafe max+1 allocator.
- [x] Run focused tests then full pytest; execute migration twice on MySQL and insert a new conversation to verify server ID.
- [x] Append task evidence immediately. Review/check diff before task commit.

## Task 2: Versioned native BM25 collection and resumable synchronization

**Files:** Create `app/services/knowledge/hybrid_store.py`, `scripts/rebuild_hybrid_index.py`, `tests/test_hybrid_store.py`; modify config/runtime/indexer/repository and `.env.example`.

**Interfaces:** `HybridRow(chunk_id, vector, text, category, content_hash)`; `HybridHit(chunk_id, score, content_hash)`; `HybridStore.search(query_vector, lexical_query, strategy, category=None, limit=50) -> list[HybridHit]`; `.upsert(rows) -> list[int]`; `.ensure_collection()`; `.delete(ids)`.

- [x] Context7: native FunctionType.BM25, chinese analyzer, sparse index, AnnSearchRequest expr, RRFRanker and collection inspection for installed versions.
- [x] Write failing adapter tests using injected SDK transport: returned results carry correct PK/hash; category applied on both ANN requests; invalid schema rejected; rebuild resume upserts same PK. Hydration test: SQL hash changed after indexing => old hit yields no evidence.
```python
def test_changed_body_rejects_old_index_hit(retriever, db_session):
    row = db_session.get(KnowledgeChunk, 7)
    row.answer = '新政策'
    row.content_hash = 'new-hash'
    db_session.commit()
    assert retriever.hydrate([HybridHit(7, 0.8, 'old-hash')]) == []
```
- [x] Run focused tests RED; implement versioned collection fields chunk_id/embedding/text/bm25/category/is_active/content_hash. Keep old collection intact; index checkpoint stored independently from legacy vector_status, comparing content hash per collection.
- [x] Rebuild CLI reads all active chunks in stable batches, embeds/upserts, marks sync only after success and removes inactive IDs with retry. Validate count/key/hash coverage before configurable cutover; refuse destructive automatic collection replacement.
- [x] Run focused/full suite, compose config; live rebuild twice and verify sample BM25/filter results.
- [x] Log and commit reviewed task.

## Task 3: Query understanding, four retrieval strategies and reranking

**Files:** Create `app/services/quality/query.py`, `app/services/quality/retrieval.py`, `app/services/quality/rerank.py`, `tests/test_quality_retrieval.py`; modify runtime/config.

**Interfaces:** `QueryUnderstanding(raw, canonical, lexical)`; `Evidence(n, chunk_id, section_path, question, answer, source_key, score)`; `QualityRetriever.hydrate(hits: list[HybridHit]) -> list[Evidence]`; `QualityRetriever.retrieve(query, strategy='hybrid_rerank', category=None) -> list[Evidence]`; `Reranker.rank(query, documents, top_n=10)`.

- [x] Context7 LangChain structured output, SiliconFlow rerank HTTP contract; normalization prompt validation uses labeled examples.
- [x] Tests RED: preserved XH-300 model and “不能” condition; normalization parse failure uses raw question; duplicate/out-of-range/nonfinite rerank results fail; four strategies choose their correct paths and cap evidence at ten.
```python
def test_bad_rerank_index_is_rejected(client):
    client.response = {'results': [{'index': 3, 'relevance_score': .9}]}
    with pytest.raises(ValueError):
        client.rank('退款', ['一条文档'], top_n=10)
```
- [x] Implement single-turn canonicalization, bounded retrieval synonyms, deterministic preservation validation. Implement HTTP rerank with timeouts and safe errors; no model fallback.
- [x] Hand-check normalization against policy/model/colloquial/negative samples; run focused/full suite. Log and commit.

## Task 4: Guarded generation, low-confidence capture and SSE citations

**Files:** Create `app/services/quality/generation.py`, `tests/test_quality_generation.py`; modify chat/business/prompts/schemas/main.

**Interfaces:** `arrange_evidence(evidence) -> list[Evidence]`; `GuardedAnswer(answer, citations, refused, reason, source)`; `KnowledgeAnswerService.answer(raw_question, conversation_key, category=None) -> GuardedAnswer`.

- [x] Context7 LangChain with_structured_output/Pydantic, FastAPI request/stream API and SQLAlchemy JSON persistence.
- [x] Tests RED: ranks 1..6 become 1,3,5,6,4,2 without renumbering; unknown evidence creates retrieval_low_conf row; sufficient=false creates self_check row; invalid/missing citation produces refusal; pool insert failure produces error with no successful done; no-tool policy branch cannot bypass knowledge gate.
```python
def test_prompt_order_keeps_numbers():
    assert [e.n for e in arrange_evidence(sample_evidence)] == [1, 3, 5, 6, 4, 2]
```
- [x] Implement buffered structured sufficiency/answer generation and citation validator. Score gate calibrated from development labels, never threshold RRF using COSINE. Fixed negative-knowledge prompt forbids unsupported commitments.
- [x] Route knowledge facts through controlled service; preserve operational tools/history. Persist full evidence with assistant message, emit citations/token/done only after gate and durable refusal insertion.
- [x] Validate pure prompts using labeled sufficient/insufficient examples, run chat/API/full regression. Log and commit.

## Task 5: At least 300 ground-truth queries and isolated model corpus

**Files:** Create `evaluation/ch04/cases.json`, `evaluation/ch04/README.md`, `evaluation/ch04/corpus/`, `scripts/validate_ch04_dataset.py`.

**Interfaces:** Each case has eval_id/bucket/difficulty/split/query/category/relevant_source_keys/required_facts/should_refuse; 300 distinct questions and stable source labels.

- [x] Build 60 distinct meaningful cases per bucket, 20 per difficulty; split 12 calibration/48 test per bucket without canonical-question leakage. Include multi-source evidence, exact/near model IDs, colloquial synonyms, conditional and unavailable facts.
- [x] Label against existing ecommerce documents plus explicitly fictional controlled model manuals in isolated evaluation corpus; never claim invented specs apply to actual products.
- [x] Run dataset validator: unique ids/query strings, all labels resolve, no empty positive truth, unknown has no relevant sources, counts and split/difficulty coverage, manual samples across all 15 bucket/difficulty cells. Prompt/data work uses this validation in place of TDD.
- [x] Log dataset totals and any label repairs; commit reviewed data.

## Task 6: Metrics, live comparison, faithfulness judge and case lifecycle CLI

**Files:** Create `app/services/quality/evaluation.py`, `scripts/evaluate_ch04.py`, `scripts/faith_cases.py`, `tests/test_quality_evaluation.py`.

**Interfaces:** `retrieval_metrics(ranked_ids, relevant_ids, ks) -> dict`; `evaluate(cases, strategies, pipeline, judge) -> dict`; reports under `evaluation/ch04/runs/<run_id>/`.

- [x] Context7 structured-output judge and MySQL upsert/transaction semantics before implementation.
- [x] Tests RED: relevant={2,4}, ranked=[9,2,4] => Recall@1=0, Recall@5=1, MRR=.5; no-evidence negatives excluded from positive retrieval denominator; refusal faithfulness N/A; judge errors excluded/recorded; only hybrid_rerank fabrications update durable cases once per question/run.
```python
def test_hand_checked_metrics():
    metrics = retrieval_metrics([9, 2, 4], {2, 4}, [1, 5])
    assert metrics == {'recall@1': 0, 'recall@5': 1, 'mrr': .5}
```
- [x] Implement statement-grounded judge and metrics, separately track refusal/error rates, candidate Recall@50 and final Recall@1/5/10; per bucket/difficulty and macro/micro aggregation with denominators.
- [x] CLI supports --fixture (synthetic), --live, --split, --output-dir, --strategy/all and --calibrate; live immutable JSON/Markdown reports include evidence snapshots, config/model/corpus fingerprints and failures. Case CLI lists/updates disposition with required explanation.
- [x] Validate fixture numerics and persistence, run all tests, then live four-strategy evaluation on test split and separate calibration. Log and commit.

## Task 7: Source routes and frontend citation/feedback interactions

**Files:** Create `app/services/quality/sources.py`, `tests/test_quality_sources.py`; modify main/frontend/src/App.jsx/frontend/src/styles.css.

**Interfaces:** Source URL resolves registered doc source_key and deterministic section anchor; frontend reads citations SSE event and binds it only to the current message.

- [x] Context7 FastAPI safe response handling; backend tests RED for registered source path and traversal rejection, unavailable source returns explicit status. Implement source routes and consistent section anchors.
- [x] Vibe Coding frontend: clickable numbered citations, source drawer original chunk/path and jump link, per-message 👍/👎 chosen/“已反馈”/locked; localStorage scoped conversation/message, no network feedback request. Ignore welcome/streaming/error messages.
- [x] Verify browser interactions and `npm run build`; regression backend tests. Frontend exempt from code review; backend routes remain reviewed.
- [x] Log concrete interaction checks and commit.

## Task 8: Integration, independent backend review and finish

**Files:** Modify README/dev-notes/ch04.md; retain generated live reports.

- [x] Run full pytest, frontend build, docker compose config, git diff --check and live migration/rebuild/chat acceptance. Include exact-model BM25, unknown-question pool record, self-check refusal, citation jump and case recurrence demonstrations.
- [x] Invoke requesting-code-review for one fresh backend reviewer; frontend excluded. Critical/important findings fixed with RED→GREEN and rerun relevant/full suite, record receiving-code-review conclusions.
- [x] Deliver actual demo commands and live four-strategy 300-case coverage reports; identify any unavailable service without substituting synthetic acceptance. Update notes after review, not retroactively.
- [x] Use finishing-a-development-branch; do not merge/push without user authorization.

## Plan self-review

Spec coverage mapped: compatibility/ledgers Task 1; index/version recovery Task 2; query/retrieval Task 3; guard/prompt/SSE Task 4; 300-case truth Task 5; metrics/faith cases Task 6; UI/source Task 7; acceptance/review Task 8. Shared Evidence numbering, source labels, numeric conversation IDs and corpus hashes are explicit interfaces. All five review-focus conditions have tests in owning tasks. No placeholder stages; missing runtime access is reported rather than replaced.

Execution complete 2026-10-01; all eight tasks verified. Independent review findings and final live measurements are recorded in dev-notes/ch04.md. Integration choice remains with the user.
