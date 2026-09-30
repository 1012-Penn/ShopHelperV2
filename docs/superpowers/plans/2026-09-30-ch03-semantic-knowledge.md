# 第三章：电商客服语义知识库实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `query_faq` 内部升级为 BGE-M3 + Milvus dense 检索，并实现可恢复的 Markdown/FAQ/历史对话知识构建，同时保持工具 JSON 契约不变。

**Architecture:** 在当前 FastAPI/SQLAlchemy 项目内增加 `app.services.knowledge` 模块。MySQL 保存权威正文和 pending 状态；SiliconFlow BGE-M3 提供 1024 维 embedding；Milvus 以 MySQL 主键为 PK upsert 向量。独立 CLI 负责 Markdown/FAQ 导入、对话抽取/全局去重与 pending 修复；`query_faq` 只调用 dense retriever。

**Tech Stack:** Python 3.10+、FastAPI、SQLAlchemy 2、MySQL、PyMilvus、Milvus standalone、OpenAI-compatible embeddings API（SiliconFlow `BAAI/bge-m3`）、LangChain `ChatOpenAI`、pytest。

**Spec:** `docs/superpowers/specs/2026-09-30-ch03-semantic-knowledge-design.md`

## Global Constraints

- “BGE-M3 使用用户提供的 SiliconFlow OpenAI-compatible embeddings 服务，模型 ID 为 `BAAI/bge-m3`。”
- “Milvus 是 dense 向量检索库，MySQL `knowledge_chunks` 是知识原文和状态的权威源。”
- “`query_faq` 的对外契约保持不变；本章不实现关键词召回、混合检索、稀疏检索或重排。”
- “`embedding_text` 固定由 `category`、`questions` 和 `answer` 拼接生成。”
- “默认 `max_chars=1200`、`overlap_chars=200`，可由调用配置覆盖；这些是目标而非硬上限。”
- “历史消息抽取按用户配置的系统 cron 周期执行；任务结束返回成功/失败退出码并输出处理数，不新增常驻 scheduler。”
- “密钥不得进入源码、README、测试夹具、开发记录或日志。”
- 具体 API 先查 Context7 与官方文档，再按实际安装版本实现；本计划不得将文档示例当作版本锁定依据。

## Review Focus

1. 单条句子或表格行比 chunk 目标更长时仍须完整保留；Task 3 加入该测试。
2. 对话含工具申请/工具结果、连续 user 消息、空回答或可识别个人信息时不能错误配对或泄露；Task 6 加入这些测试。
3. Milvus upsert 成功但 MySQL 状态回填前崩溃时，重跑不能重复向量；Task 5 验证该故障点。
4. 低于相似度阈值或 Milvus 返回 MySQL 无对应记录时不能伪报命中；Task 7 验证契约和恢复行为。
5. FAQ 同义问题与无关问题的阈值要用标注集校准；Task 8 执行检索评估并记录阈值与指标。

---

## 文件与接口地图

- `app/config.py`：新增 embeddings/Milvus/chunk/retrieval 参数与校验。
- `app/db/models.py`、`app/db/session.py`：知识块、暂存问答和抽取游标 ORM；沿用现有 `create_all` 增量建表方式。
- `app/services/knowledge/chunking.py`：Markdown 层级解析、递归分块和表格处理。
- `app/services/knowledge/content.py`：知识草稿类型、embedding 文本和稳定指纹。
- `app/services/knowledge/repository.py`：MySQL 知识块、暂存候选和 checkpoint 操作；`load_turns_after(last_message_id: int, limit: int) -> tuple[list[ConversationTurn], int]`、`stage_batch_and_advance(candidates: list[tuple[int, ExtractedCandidate]], last_message_id: int, run_id: str) -> None`、`promote_staged(run_id: str) -> tuple[int, int]`（去重数、新增数）。
- `app/services/knowledge/embeddings.py`：OpenAI Python client 封装；`embed_documents(texts: list[str]) -> list[list[float]]`、`embed_query(text: str) -> list[float]`。
- `app/services/knowledge/vector_store.py`：PyMilvus adapter；`ensure_collection() -> None`、`upsert(rows: list[VectorRow]) -> list[int]`、`search(vector: list[float], limit: int) -> list[VectorHit]`、`delete(ids: list[int]) -> None`。
- `app/services/knowledge/indexer.py`：原文先入 MySQL pending，后生成 embedding/upsert/backfill；`sync_pending(batch_size: int) -> SyncSummary`。
- `app/services/knowledge/conversations.py`：会话消息批处理、脱敏、结构化抽取、staging 和全局去重；`run(batch_size: int) -> ExtractionSummary`。
- `app/services/knowledge/retriever.py`：`search(query: str) -> list[FAQHit]`，阈值过滤并回查 MySQL。
- `app/tools/business.py`、`app/services/chat.py`、`app/main.py`：注入 FAQ retriever，保持 LangChain 工具 Schema 和响应契约。
- `scripts/build_knowledge.py`、`scripts/extract_conversation_knowledge.py`、`scripts/sync_knowledge_vectors.py`：三条离线命令。
- `knowledge_docs/`：版本控制的通用电商客服 Markdown 基线语料，不含虚构店铺费率或 SLA。
- `docker-compose.yml`、`.env.example`、`README.md`：Milvus 服务、参数说明、cron 与演示命令。
- `tests/test_knowledge_*.py`、`tests/fixtures/faq_cases.json`：分块、存储、恢复、抽取和标注评估。
- `dev-notes/ch03.md`：每个阶段完成即追加原话、产出/评审、纠偏、翻车记录。

类型定义：`VectorRow(chunk_id: int, vector: list[float])`；`VectorHit(chunk_id: int, score: float)`，其中 score 是 COSINE similarity；`SyncSummary(pending_before: int, vectorized: int, failed: int)`；`ConversationTurn(user_message_id: int, assistant_message_id: int, user_text: str, assistant_text: str)`；Pydantic `ExtractedCandidate(source_turn_index: int, category: str, questions: list[str], answer: str)` 与 `ExtractionBatch(candidates: list[ExtractedCandidate])` 是模型结构化响应；应用校验 turn index 后把来源消息 ID 加入 staging；`ExtractionSummary(messages_read: int, staged_pairs: int, deduped: int, inserted: int, skipped_tool_messages: int, last_message_id: int)`；`FAQHit(question: str, answer: str, category: str, score: float)`。

## 任务 1：配置与运行依赖

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`
- Modify: `.env.example`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces `Settings` fields: `embedding_api_key`, `embedding_api_base`, `embedding_model`, `milvus_uri`, `milvus_collection`, `knowledge_max_chars`, `knowledge_overlap_chars`, `faq_top_k`, `faq_min_similarity`, `knowledge_batch_size`。
- Defaults: model=`BAAI/bge-m3`, base=`https://api.siliconflow.cn/v1`, collection=`knowledge`, max chars=1200, overlap chars=200, Top-K=5, initial minimum cosine similarity=0.40 (Task 8 replaces it with the measured labeled-set threshold). API key and `MILVUS_URI` have no implicit secret/default credential; accessing knowledge features without them raises a safe configuration error.
- `Settings.require_knowledge() -> None` validates knowledge-only credentials, leaving existing chat-only settings behavior unchanged.

**Test case:**

```python
def test_knowledge_settings_validate_chunking_and_retrieval_limits():
    settings = Settings.from_env({
        "MODEL": "chat-model", "API_KEY": "chat-key", "BASE_URL": "https://chat.example/v1",
        "DATABASE_URL": "sqlite://", "EMBEDDING_API_KEY": "embed-key",
        "MILVUS_URI": "http://localhost:19530", "KNOWLEDGE_MAX_CHARS": "1200",
        "KNOWLEDGE_OVERLAP_CHARS": "200", "FAQ_TOP_K": "5",
    })
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_api_base == "https://api.siliconflow.cn/v1"
    assert settings.faq_top_k == 5
```

- [ ] **Step 1: Write failing settings tests** for the default non-secret values, key/base/model overrides, invalid chunk sizes (`max_chars <= overlap_chars`), Top-K outside 1..20, and missing credentials when knowledge settings are required.
- [ ] **Step 2: Run the red tests.** `python3 -m pytest tests/test_config.py -q`; expected: failures because knowledge fields and validation do not exist.
- [ ] **Step 3: Add direct runtime dependencies** `openai>=2.45,<4` and `pymilvus>=3.0.2,<4`. The installed `langchain-openai==1.6.3` requires `openai>=2.45,<4`; package index reports PyMilvus 3.0.2 as current. Keep existing broad project dependency policy; do not install or load local FlagEmbedding weights.
- [ ] **Step 4: Add Settings fields and env parsing.** Keep `Settings.from_env` behavior for existing chat configuration; `Settings.require_knowledge()` validates that the embedding key and Milvus URI are present only when constructing knowledge services.
- [ ] **Step 5: Add `.env.example` names and safe example values.** Add `EMBEDDING_API_BASE=https://api.siliconflow.cn/v1`, `EMBEDDING_MODEL=BAAI/bge-m3`, `EMBEDDING_API_KEY=`, `MILVUS_URI=http://127.0.0.1:19530`, `MILVUS_COLLECTION=knowledge`, `KNOWLEDGE_MAX_CHARS=1200`, `KNOWLEDGE_OVERLAP_CHARS=200`, `FAQ_TOP_K=5`, `FAQ_MIN_SIMILARITY=0.40`, `KNOWLEDGE_BATCH_SIZE=32`; never copy the user's key.
- [ ] **Step 6: Run settings and existing config tests.** `python3 -m pytest tests/test_config.py -q`; expected: all pass with no secret in captured output.
- [ ] **Step 7: Commit.** `git add pyproject.toml app/config.py .env.example tests/test_config.py && git commit -m "feat: configure semantic knowledge services"`.

## 任务 2：MySQL ORM schema

**Files:**
- Modify: `app/db/models.py`
- Modify: `app/db/session.py` only if model registration is needed
- Test: `tests/test_models.py`

**Interfaces:**
- `KnowledgeChunk`: `id`, unique `source_key`, `category`, JSON `questions`, `answer`, `embedding_text`, JSON `chapter_path`, `content_type`, `is_critical`, nullable self-FKs `previous_chunk_id`/`next_chunk_id`, nullable `vector_id`, `vector_status`, `content_hash`, timestamps.
- `KnowledgeQAStaging`: source user/assistant message IDs, category, JSON questions, answer, canonical fingerprint, run ID, created time; unique fingerprint/source guard.
- `KnowledgeExtractionCursor`: named cursor primary key, `last_message_id`, updated time.
- Status strings are exactly `pending` and `vectorized`; failure diagnostics do not expose API credentials.

**Test case:**

```python
def test_knowledge_chunk_round_trips_json_fields_and_neighbor_links(db_session):
    chunk = KnowledgeChunk(
        source_key="doc:shipping:0", category="配送", questions=["运费是多少"], answer="以结算页为准。",
        embedding_text="配送\n运费是多少\n以结算页为准。", chapter_path=["配送", "运费"],
        content_type="policy", is_critical=False, vector_status="pending",
    )
    db_session.add(chunk)
    db_session.flush()
    assert db_session.get(KnowledgeChunk, chunk.id).questions == ["运费是多少"]
    assert chunk.vector_id is None and chunk.vector_status == "pending"
```

- [ ] **Step 1: Write failing ORM tests** asserting metadata fields, JSON question/path round-trip, nullable neighbor links, unique source/fingerprint constraints, and that a fresh SQLite schema contains the three knowledge tables.
- [ ] **Step 2: Run red tests.** `python3 -m pytest tests/test_models.py -q`; expected: import/attribute failures for knowledge entities.
- [ ] **Step 3: Add ORM models.** Use SQLAlchemy 2 `Mapped` annotations and database-portable types already used by the repository. Keep `FAQ`, `Conversation`, `Message`, and `Ticket` unchanged.
- [ ] **Step 4: Verify DDL portability.** Add a test compiling `KnowledgeChunk` DDL for MySQL and assert no unique index is created over a `TEXT` column; fingerprint and source keys use bounded strings.
- [ ] **Step 5: Run model tests.** `python3 -m pytest tests/test_models.py -q`; expected: all pass on SQLite and MySQL DDL compilation.
- [ ] **Step 6: Commit.** `git add app/db/models.py app/db/session.py tests/test_models.py && git commit -m "feat: add knowledge persistence models"`.

## 任务 3：Markdown chunker 与通用电商客服语料

**Files:**
- Create: `app/services/knowledge/__init__.py`
- Create: `app/services/knowledge/content.py`
- Create: `app/services/knowledge/chunking.py`
- Create: `knowledge_docs/01-商品与下单.md`
- Create: `knowledge_docs/02-配送与运费.md`
- Create: `knowledge_docs/03-退换货与退款.md`
- Create: `knowledge_docs/04-支付与售后.md`
- Test: `tests/test_knowledge_chunking.py`

**Interfaces:**
- `ChunkDraft(source_key: str, category: str, questions: list[str], answer: str, chapter_path: list[str], content_type: str, is_critical: bool, previous_source_key: str | None, next_source_key: str | None)`。
- `split_markdown(source: str, markdown: str, *, max_chars: int = 1200, overlap_chars: int = 200) -> list[ChunkDraft]`。
- `build_embedding_text(category: str, questions: list[str], answer: str) -> str` serializes only those three fields in deterministic order.
- Generated corpus is system-agnostic. It states that store-specific price, shipping fee, service level, warranty and regional exceptions must be read from actual product/order/checkout/published policy; it contains no invented guaranteed number.

**Test case:**

```python
def test_oversize_sentence_is_kept_whole_and_table_headers_repeat():
    long_sentence = "运费规则" + "非常重要" * 400 + "。"
    markdown = "# 配送\n\n" + long_sentence + "\n\n|区间|规则|\n|---|---|\n|A|甲|\n|B|乙|"
    chunks = split_markdown("shipping.md", markdown, max_chars=20, overlap_chars=5)
    assert any(long_sentence in chunk.answer for chunk in chunks)
    table_chunks = [chunk.answer for chunk in chunks if "|区间|规则|" in chunk.answer]
    assert len(table_chunks) == 2
    assert all("|---|---|" in chunk for chunk in table_chunks)
```

- [ ] **Step 1: Write failing chunking tests** for nested heading path and parent category, product FAQ question preservation, a paragraph exceeding the target, a single sentence longer than target, Chinese/English sentence marks, overlap ending only at a complete sentence, table rows split with header/separator repeated, code fences kept balanced, deterministic source keys, and embedding text excluding metadata.
- [ ] **Step 2: Run red tests.** `python3 -m pytest tests/test_knowledge_chunking.py -q`; expected: `split_markdown` and `ChunkDraft` are unavailable.
- [ ] **Step 3: Implement structural Markdown parsing** with heading stack and typed segments for prose/tables/fenced code. Do not split a sentence or table row solely to meet the soft char target.
- [ ] **Step 4: Implement recursive splitting and sentence-safe overlap.** Target `max_chars=1200`, overlap goal `200`; choose the nearest preceding complete sentence boundary; include at least one prior whole sentence when available, never cut mid-sentence.
- [ ] **Step 5: Write production-baseline ecommerce documents.** Cover product selection/order, delivery/fees, returns/refunds, payments and after-sales. Use policy-safe wording and retain section titles for knowledge questions/category paths.
- [ ] **Step 6: Run chunking tests.** `python3 -m pytest tests/test_knowledge_chunking.py -q`; expected: every boundary and metadata assertion passes.
- [ ] **Step 7: Commit.** `git add app/services/knowledge knowledge_docs tests/test_knowledge_chunking.py && git commit -m "feat: add structured ecommerce knowledge chunking"`.

## 任务 4：MySQL 知识写入与来源幂等

**Files:**
- Create: `app/services/knowledge/repository.py`
- Create: `app/services/knowledge/indexer.py`
- Test: `tests/test_knowledge_indexer.py`
- Test: `tests/test_models.py`

**Interfaces:**
- `KnowledgeRepository(session_factory)` provides `upsert_drafts(drafts: list[ChunkDraft]) -> list[int]`, `pending(limit: int) -> list[KnowledgeChunk]`, `mark_vectorized(chunk_id: int, vector_id: int) -> None`, `load_by_ids(ids: list[int]) -> list[KnowledgeChunk]`, `load_turns_after(last_message_id: int, limit: int) -> tuple[list[ConversationTurn], int]`, `stage_batch_and_advance(candidates: list[tuple[int, ExtractedCandidate]], last_message_id: int, run_id: str) -> None`, and `promote_staged(run_id: str) -> tuple[int, int]`。
- FAQ source mapping: `FAQ.id -> source_key=f"faq:{faq.id}"`, `questions=[faq.question]`, `answer=faq.answer`, `category=faq.category`, `content_type="product_faq"`.
- Document source keys are stable for a source/section/block ordinal. Same source and same content is a no-op; changed body updates same row and marks it pending.

**Test case:**

```python
def test_same_source_update_resets_vector_state_and_keeps_primary_key(repository, drafts):
    chunk_id = repository.upsert_drafts(drafts)[0]
    repository.mark_vectorized(chunk_id, chunk_id)
    changed = [replace(drafts[0], answer="新版运费说明。")]
    assert repository.upsert_drafts(changed) == [chunk_id]
    row = repository.pending(limit=10)[0]
    assert row.answer == "新版运费说明。"
    assert row.vector_status == "pending"
```

- [ ] **Step 1: Write failing repository tests** for FAQ-to-draft mapping, unique source upsert, content update resetting vector state, pending ordering, stable neighbor pointers, and empty import idempotency.
- [ ] **Step 2: Run red tests.** `python3 -m pytest tests/test_knowledge_indexer.py -q`; expected: repository import failure.
- [ ] **Step 3: Implement deterministic content hash and source mapping** in `content.py`; normalize Unicode, whitespace, and punctuation only for fingerprints, not for stored answer text.
- [ ] **Step 4: Implement transactional SQLAlchemy repository methods.** Commit source rows as pending in short transactions before any network request; use `session_factory.begin()`.
- [ ] **Step 5: Link previous/next chunk IDs** after inserting the full source draft list; verify relationships remain null at list boundaries.
- [ ] **Step 6: Run repository, model, and existing seed tests.** `python3 -m pytest tests/test_knowledge_indexer.py tests/test_models.py -q`; expected: all pass, including existing FAQ seed idempotency.
- [ ] **Step 7: Commit.** `git add app/services/knowledge/content.py app/services/knowledge/repository.py app/services/knowledge/indexer.py tests/test_knowledge_indexer.py tests/test_models.py && git commit -m "feat: persist knowledge chunks idempotently"`.

## 任务 5：SiliconFlow embeddings 与 Milvus pending 恢复

**Files:**
- Create: `app/services/knowledge/embeddings.py`
- Create: `app/services/knowledge/vector_store.py`
- Modify: `app/services/knowledge/indexer.py`
- Modify: `docker-compose.yml`
- Test: `tests/test_knowledge_vectors.py`

**Interfaces:**
- `EmbeddingClient.embed_documents(texts: list[str]) -> list[list[float]]`; `EmbeddingClient.embed_query(text: str) -> list[float]`。
- `VectorRow(chunk_id: int, vector: list[float])`; `VectorHit(chunk_id: int, score: float)`。
- `MilvusKnowledgeStore.ensure_collection() -> None`; `.upsert(rows: list[VectorRow]) -> list[int]`; `.search(vector: list[float], limit: int) -> list[VectorHit]`。
- `KnowledgeIndexer.sync_pending(batch_size: int) -> SyncSummary` embeds pending text, validates dimensions/order, Milvus-upserts by MySQL ID, then backfills returned ID/status.

**Test case:**

```python
def test_rerun_after_milvus_success_before_mysql_backfill_upserts_same_id(indexer, vector_store, repository):
    vector_store.fail_after_upsert_once = True
    first = indexer.sync_pending(batch_size=8)
    assert first.failed == 1
    chunk_id = repository.pending(limit=1)[0].id
    indexer.sync_pending(batch_size=8)
    assert vector_store.ids == {chunk_id}
    row = repository.load_by_ids([chunk_id])[0]
    assert row.vector_id == chunk_id and row.vector_status == "vectorized"
```

- [ ] **Step 1: Query current Context7 and official docs before coding** for OpenAI Python and PyMilvus. Confirmed APIs: `OpenAI(base_url=..., api_key=...)`, `client.embeddings.create(model=..., input=...)`, `MilvusClient.create_schema/create_collection`, `upsert` response `ids`, `search` hits `id/distance`, and `delete(ids=...)`. SiliconFlow documents `BAAI/bge-m3` at 8192 tokens; successful configured call returned 1024 dimensions. Keep dependency bounds `openai>=2.45,<4` and `pymilvus>=3.0.2,<4`; record installed versions and docs access date in this task note without secrets.
- [ ] **Step 2: Write failing embedding tests** with an injected fake OpenAI client, asserting batch input order, `response.data[*].index` ordering, 1024 dimension validation and safe error propagation.
- [ ] **Step 3: Write failing Milvus/indexer tests** asserting known primary key upsert, COSINE collection, Top-K hit parsing, return-ID backfill and state transitions.
- [ ] **Step 4: Run red tests.** `python3 -m pytest tests/test_knowledge_vectors.py -q`; expected: missing client/store/sync methods.
- [ ] **Step 5: Implement embeddings and Milvus adapters** with dependency injection. Ensure collection schema uses Int64 `chunk_id` PK and 1024-d FloatVector; upsert rows use the MySQL primary key and return IDs in corresponding row order.
- [ ] **Step 6: Implement sync ordering and failure semantics.** MySQL pending commit precedes embedding; Milvus upsert precedes SQL vectorized status. Catch per-row failures, increment `SyncSummary.failed`, continue other rows, and leave the failed row pending so the next run upserts the same ID. CLI exits nonzero when any row failed.
- [ ] **Step 7: Add official Milvus standalone service and persistent volumes** to compose, preserving existing MySQL behavior. Configure URI and health/readiness checks without exposing credentials.
- [ ] **Step 8: Test both interruption points.** Inject failure before Milvus upsert and after successful upsert before MySQL state update; rerun sync and assert one Milvus PK per chunk, matching `vector_id`, status vectorized.
- [ ] **Step 9: Run vector tests and compose validation.** `python3 -m pytest tests/test_knowledge_vectors.py -q` and `docker compose config`; expected: tests pass and valid compose YAML.
- [ ] **Step 10: Commit.** `git add app/services/knowledge/embeddings.py app/services/knowledge/vector_store.py app/services/knowledge/indexer.py docker-compose.yml tests/test_knowledge_vectors.py pyproject.toml && git commit -m "feat: sync knowledge vectors idempotently"`.

## 任务 6：客服对话挖掘、脱敏、暂存和全局去重

**Files:**
- Create: `app/services/knowledge/privacy.py`
- Create: `app/services/knowledge/conversations.py`
- Modify: `app/db/models.py`
- Test: `tests/test_knowledge_conversations.py`

**Interfaces:**
- `redact_customer_data(text: str) -> str` masks mainland Chinese mobile numbers, email addresses, 18-digit identity numbers, and order/ticket identifiers before an external model call; it never modifies persisted source messages.
- `ConversationKnowledgeExtractor(model, repository)` exposes `run(batch_size: int) -> ExtractionSummary`。
- The injected model implements `extract_pairs(turns: list[ConversationTurn]) -> list[ExtractedCandidate]`; production adapter wraps `ChatOpenAI.with_structured_output(ExtractionBatch)` and tests use a deterministic fake.

**Test case:**

```python
def test_extractor_redacts_contact_data_and_skips_tool_messages(extractor, fake_model):
    result = extractor.run(batch_size=20)
    sent_text = " ".join(fake_model.last_input)
    assert "13800138000" not in sent_text
    assert "user@example.com" not in sent_text
    assert result.staged_pairs == 1
    assert result.skipped_tool_messages >= 1
```

- [ ] **Step 1: Write failing privacy and turn-selection tests** for phone/email/ID/order redaction, tool messages omitted, assistant tool-call request omitted, user + final assistant answer pair retained, consecutive user messages not mispaired, and empty answers skipped.
- [ ] **Step 2: Write failing checkpoint/staging tests** for two batches, repeat run idempotency, exception without cursor advancement, duplicate Q&A across batches, and duplicate against existing `knowledge_chunks`.
- [ ] **Step 3: Run red tests.** `python3 -m pytest tests/test_knowledge_conversations.py -q`; expected: missing extractor/privacy functions.
- [ ] **Step 4: Implement pure privacy redaction** and assert sanitized text is the only text passed to the fake/external extractor; store extracted generic content, never raw transcript in staging answer fields.
- [ ] **Step 5: Implement message batching by `Message.id`** using a durable cursor. Pair each user message with the next assistant final answer only if it occurs before another user message; skip unmatched turns and assistant tool-call requests. Include stable per-batch `source_turn_index` values in the prompt, validate indexes are in range, and map selected candidates back to local user/assistant IDs. Advance cursor only after all candidates for that batch commit into staging (also advance when valid pairs produce zero candidates).
- [ ] **Step 6: Implement structured extraction adapter** using documented LangChain `with_structured_output` and strict empty/invalid candidate validation; on parse/provider error leave checkpoint unchanged.
- [ ] **Step 7: Implement stage-level and global canonical deduplication.** Normalize category/question list/answer, hash deterministically, compare current run plus existing knowledge, then create only unseen pending `KnowledgeChunk` rows.
- [ ] **Step 8: Run dialogue tests and all existing chat/model tests.** `python3 -m pytest tests/test_knowledge_conversations.py tests/test_chat.py tests/test_models.py -q`; expected: all pass and no source message contents appear in logs.
- [ ] **Step 9: Commit.** `git add app/services/knowledge/privacy.py app/services/knowledge/conversations.py app/db/models.py tests/test_knowledge_conversations.py && git commit -m "feat: extract deduplicated knowledge from conversations"`.

## 任务 7：FAQ dense retriever 与契约兼容

**Files:**
- Create: `app/services/knowledge/retriever.py`
- Modify: `app/tools/business.py`
- Modify: `app/services/chat.py`
- Modify: `app/main.py`
- Modify: `tests/test_tools.py`
- Create: `tests/test_knowledge_retriever.py`
- Modify: `tests/fixtures/faq_cases.json`

**Interfaces:**
- `FAQHit(question: str, answer: str, category: str, score: float)`。
- `KnowledgeRetriever(searcher, repository, top_k: int, min_similarity: float).search(query: str) -> list[FAQHit]`。
- `build_tools(session_factory, conversation_id, faq_retriever)` registers `query_faq(query: str) -> str`; only internal dependency changes.
- `ChatService(session_factory, model_factory, tool_runner_factory, faq_retriever)` stores one app-scoped retriever and binds it into each tools list; `create_app(chat_service=...)` remains usable for isolated API tests.

**Test case:**

```python
def test_query_faq_preserves_tool_schema_and_json_contract(tools_by_name, fake_retriever):
    faq = tools_by_name["query_faq"]
    assert set(faq.get_input_schema().model_fields) == {"query"}
    assert json.loads(faq.invoke({"query": "邮费是多少"})) == {
        "matched": True,
        "items": [{"question": "运费是多少？", "answer": "以结算页显示为准。", "category": "配送"}],
    }
```

- [ ] **Step 1: Write failing retriever tests** for query embedding, ordered MySQL hydration, category/question mapping, Top-K cap, threshold filtering and dangling Milvus IDs.
- [ ] **Step 2: Rewrite FAQ contract tests first.** Keep exact name and input schema; expect same JSON keys/types and no-match message while hits now come from injected retriever instead of `FAQ.question LIKE`.
- [ ] **Step 3: Run red tests.** `python3 -m pytest tests/test_knowledge_retriever.py tests/test_tools.py -q`; expected: constructor signature/behavior mismatch and old LIKE assertion fails.
- [ ] **Step 4: Implement retriever.** Embed the query, perform Milvus COSINE Top-K, discard hits below `min_similarity`, fetch MySQL rows by primary key, preserve vector ranking, ignore missing SQL rows, and return at most five items.
- [ ] **Step 5: Inject runtime dependencies.** Build one embedding client/store/retriever in `app.main` and inject it into `ChatService` once; `ChatService` passes that instance into each `build_tools` call. Do not change tool descriptions to promise keyword behavior.
- [ ] **Step 6: Update FAQ marked cases** with query, expected source, expected answer phrase and hit/miss label. Include “邮费是多少” and answer assertion against the generated delivery/fee document; include unrelated negative queries.
- [ ] **Step 7: Run retriever, tools, chat, and API tests.** `python3 -m pytest tests/test_knowledge_retriever.py tests/test_tools.py tests/test_chat.py tests/test_api.py -q`; expected: full tool/API compatibility passes.
- [ ] **Step 8: Commit.** `git add app/services/knowledge/retriever.py app/tools/business.py app/services/chat.py app/main.py tests/test_tools.py tests/test_knowledge_retriever.py tests/fixtures/faq_cases.json && git commit -m "feat: retrieve faq answers with dense vectors"`.

## 任务 8：CLI、运行手册、真实服务验收与最终评估

**Files:**
- Create: `scripts/build_knowledge.py`
- Create: `scripts/extract_conversation_knowledge.py`
- Create: `scripts/sync_knowledge_vectors.py`
- Modify: `README.md`
- Modify: `tests/test_api.py` or create `tests/test_knowledge_cli.py`
- Modify: `dev-notes/ch03.md`

**Interfaces:**
- `python3 -m scripts.build_knowledge --source-dir knowledge_docs` imports Markdown + legacy FAQ, then syncs pending vectors.
- `python3 -m scripts.extract_conversation_knowledge --batch-size 100` stages/deduplicates conversation Q&A and syncs pending vectors.
- `python3 -m scripts.sync_knowledge_vectors --batch-size 32` repairs pending rows only.
- All commands print aggregate counts and return nonzero on a failed batch; no key/token/exception traceback is printed.

**Test case:**

```python
def test_vector_cli_reports_safe_nonzero_failure(monkeypatch, capsys):
    class FailingIndexer:
        def sync_pending(self, batch_size):
            raise RuntimeError("sk_demo_secret upstream detail")

    monkeypatch.setattr("scripts.sync_knowledge_vectors.build_indexer", lambda: FailingIndexer())
    assert sync_knowledge_vectors.main(["--batch-size", "32"]) == 1
    output = capsys.readouterr().out
    assert "sk_demo_secret" not in output
    assert "pending" in output
```

- [ ] **Step 1: Write failing CLI tests** for argparse values, missing settings returning nonzero safely, extraction invoking staging/finalization/sync once, vector CLI touching only pending records, and summary output excluding secrets.
- [ ] **Step 2: Run red tests.** `python3 -m pytest tests/test_knowledge_cli.py -q`; expected: script module/entrypoint failures.
- [ ] **Step 3: Implement three CLI entrypoints** with injectable service factories and `main(argv=None) -> int`; use short transactions and close SQL/Milvus clients in `finally` blocks.
- [ ] **Step 4: Add README setup/demo commands** for installing extras, copying env names without a real key, `docker compose up -d db milvus`, build knowledge, extraction, pending repair, cron example, and the FAQ chat acceptance query.
- [ ] **Step 5: Run focused CLI tests.** `python3 -m pytest tests/test_knowledge_cli.py -q`; expected: all pass without external service or secret.
- [ ] **Step 6: Run full automated suite.** `python3 -m pytest -q`; expected: existing and new tests pass.
- [ ] **Step 7: Run deployment validation.** `docker compose config`; if Docker is available, start MySQL/Milvus, run build CLI twice, run the FAQ evaluation set, inject/recover pending-vector interruption, and inspect counts/IDs without printing credentials.
- [ ] **Step 8: Calibrate similarity threshold** using the fixed fixture grid; write selected `FAQ_MIN_SIMILARITY`, hit@5, answer/source correctness, negative false-positive count and any unsupported cases into README and dev note. Do not claim acceptance if the labeled set cannot meet its contract.
- [ ] **Step 9: Update `dev-notes/ch03.md` immediately** with this task's user quote, outputs/test counts, corrections, errors/rework and review state.
- [ ] **Step 10: Commit.** `git add scripts/build_knowledge.py scripts/extract_conversation_knowledge.py scripts/sync_knowledge_vectors.py README.md tests/test_knowledge_cli.py tests/fixtures/faq_cases.json dev-notes/ch03.md && git commit -m "feat: add knowledge build and recovery commands"`.

## Final branch review and finish

- Read the complete branch diff against its implementation base and check the accepted spec line by line.
- Run `git diff --check`, `python3 -m pytest -q`, `docker compose config`, the labeled semantic evaluation and the pending-interruption recovery demonstration; capture outputs in the same task notes.
- Perform a fresh-context whole-branch code review. Fix Critical/Important findings with RED→GREEN tests and rerun the complete suite; log Minor findings and defer them explicitly.
- Complete `dev-notes/ch03.md` with review conclusion and finish stage; then use `superpowers:finishing-a-development-branch` to report integration options without merging/pushing unless the user asks.
