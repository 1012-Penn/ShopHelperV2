# Chapter 2 Function Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在客服 SSE 聊天中接入可持久化的单次 LangChain Function Calling，并提供 MySQL 演示环境。

**Architecture:** FastAPI 路由将聊天编排服务产生的事件编码为 SSE。SQLAlchemy 管理 FAQ、会话、消息和工单；工具注册/执行层通过 LangChain `@tool` 暴露五个业务工具，聊天服务只执行模型返回的一个工具调用并回灌结果。

**Tech Stack:** Python 3.10+、FastAPI、SQLAlchemy 2.0、MySQL 8、Docker Compose、LangChain、langchain-openai、PyMySQL、python-dotenv、pytest、httpx。

**Spec:** `docs/superpowers/specs/2026-09-23-ch02-function-calling-design.md`

## Global Constraints

- HTTP 聊天入口继续使用 `POST /api/v1/chat/stream`。
- 工具限于 `query_order`、`query_product`、`query_logistics`、`query_faq`、`create_ticket` 五项。
- 订单、商品和物流数据仅在工具内随机生成；不得新增这三类数据表或假接真实 API。
- FAQ 查询只对问题字段执行 SQL `LIKE`；“邮费是多少”漏召回是预期。
- 模型最多触发一次工具调用；多个调用申请拒绝执行；不得实现 Agent loop。
- MySQL 持久化 `faq`、`conversations`、`messages`、`tickets` 四表；ORM 为 schema 来源，seed 可重复运行。
- 工单会话 ID 从聊天服务绑定的上下文取得，不能由模型传入。
- 工具执行和模型调用期间不持有数据库事务；错误响应不得泄露密钥、堆栈或内部异常详情。
- Function calling 及服务端逻辑遵循 TDD；聊天页按用户要求采用 Vibe Coding 直接实现，不给聊天页单独加 TDD/code-review gate。
- 每完成阶段更新 `dev-notes/ch02.md`，记录决定、测试、演示或阻塞。

## Review Focus

1. FAQ 输入是完整中文问句而非关键词时，`LIKE` 仍按字面规则工作；命中“退货政策是什么”，漏掉“邮费是多少”。归 Task 3 FAQ 测试。
2. 模型返回未知工具、无效参数或多个工具申请时，不运行任意工具且不伪造成功记录。归 Task 4 服务测试。
3. 工具超时或可重试异常时，重试次数有限；最终错误作为 tool 结果回灌，不把未完成工单说成已创建。归 Task 3/4 测试。
4. 客户端复用 `conversation_id` 时记录按顺序追加；工具申请和结果的 `tool_call_id` 一致。归 Task 2/4 持久化测试。
5. 模型或数据库在 SSE 开始后报错时返回可展示的 `error` 事件，已有的成功消息仍可查询。归 Task 5 API 测试。

---

## File Map

- `pyproject.toml`：依赖、pytest 配置和开发工具配置。
- `.gitignore`：忽略虚拟环境、缓存、`.env`、覆盖率文件。
- `.env.example`：列出模型与数据库配置名，不包含密钥。
- `app/config.py`：从环境读取模型与数据库配置。
- `app/db/base.py`、`app/db/session.py`：ORM 基类、engine 和 Session factory。
- `app/db/models.py`：四个 SQLAlchemy ORM 模型及关系。
- `app/schemas.py`：聊天请求和事件 payload schema。
- `app/prompts.py`：客服系统提示词。
- `app/tools/registry.py`：工具注册表、Schema 校验、超时、重试和错误转换。
- `app/tools/business.py`：五个 `@tool` 定义及演示业务结果。
- `app/services/chat.py`：模型工具选择、持久化、单次执行、结果回灌和 token 事件。
- `app/main.py`：FastAPI 路由、SSE 编码、静态聊天页面挂载。
- `app/static/index.html`：现有入口的聊天页、SSE 解析、状态帧和工具徽章。
- `scripts/seed.py`：幂等 FAQ seed 命令。
- `docker-compose.yml`：MySQL 8 服务及健康检查。
- `tests/conftest.py`：临时 SQLite 数据库、Session 和 fake model fixture。
- `tests/test_config.py`、`tests/test_models.py`、`tests/test_tools.py`、`tests/test_chat.py`、`tests/test_api.py`：分层验证。
- `tests/fixtures/faq_cases.json`：FAQ 命中与预期漏召回标注样例。
- `README.md`：本地启动、演示和验证命令。
- `dev-notes/ch02.md`：阶段记录与 FAQ 漏召回记录。

## Task 1: FastAPI 项目骨架与配置

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `app/__init__.py`, `app/config.py`, `app/main.py`, `tests/test_config.py`
- Modify: `dev-notes/ch02.md`

**Interfaces:**
- Produces `Settings.from_env(environ: Mapping[str, str] | None = None) -> Settings`，字段为 `model`、`api_key`、`base_url`、`database_url`、`tool_timeout_seconds=5`、`tool_max_retries=2`。
- `pyproject.toml` 声明运行依赖 FastAPI、SQLAlchemy、PyMySQL、LangChain、langchain-openai、python-dotenv；开发依赖 pytest 和 httpx。
- Produces FastAPI `app`，`GET /health` 返回 `{"status":"ok"}`。

- [ ] **Step 1: 写缺配置、默认值和健康检查失败的测试**

```python
def test_settings_requires_model_key_and_base_url():
    with pytest.raises(ValueError, match="MODEL"):
        Settings.from_env({"DATABASE_URL": "sqlite:///test.db"})


def test_settings_reads_explicit_values_and_tool_defaults():
    settings = Settings.from_env({"MODEL": "demo", "API_KEY": "secret-from-env", "BASE_URL": "https://model.test/v1", "DATABASE_URL": "sqlite:///test.db"})
    assert settings.model == "demo"
    assert settings.tool_timeout_seconds == 5
    assert settings.tool_max_retries == 2


def test_health_endpoint_returns_ok():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: 确认测试红灯**

Run: `python3 -m pytest tests/test_config.py -q`
Expected: FAIL，因为 `app.config.Settings` 和 FastAPI `app` 尚未定义。

- [ ] **Step 3: 写配置及健康检查的最小实现**

配置优先读取项目 `.env`；缺少密钥时仅回退读取 `/root/.env` 中的 `DEEPSEEK_API_KEY`，绝不打印密钥。读取 `MODEL`、`API_KEY`（兼容 `DEEPSEEK_API_KEY`）、`BASE_URL`、`DATABASE_URL`；模型配置缺项明确抛 `ValueError`。加默认的工具超时与重试上限。添加 `GET /health`。

- [ ] **Step 4: 验证配置和 API**

Run: `python3 -m pytest tests/test_config.py -q`
Expected: 必填配置缺失时报对应字段错误；完整环境映射产生正确 Settings。再运行 `python3 -m pytest -q`。

- [ ] **Step 5: 提交阶段产物**

```bash
git add pyproject.toml .gitignore .env.example app tests/test_config.py dev-notes/ch02.md
git commit -m "feat: scaffold customer support API"
```

## Task 2: ORM 数据层、MySQL 与 FAQ seed

**Files:**
- Create: `app/db/__init__.py`, `app/db/base.py`, `app/db/session.py`, `app/db/models.py`, `scripts/seed.py`, `docker-compose.yml`, `tests/conftest.py`, `tests/test_models.py`
- Modify: `pyproject.toml`, `app/config.py`, `tests/test_config.py`, `dev-notes/ch02.md`

**Interfaces:**
- Consumes `Settings.database_url` from Task 1.
- Produces `Settings.database_url_from_env(environ: Mapping[str, str] | None = None) -> str`, which reads the configured database URL without requiring model credentials; the seed CLI uses this entry point.
- Produces `Base`、`make_engine(url: str)`、`make_session_factory(engine)`、`create_tables(engine)`。
- Produces ORM `FAQ`、`Conversation`、`Message`、`Ticket`；conversation 主键为字符串 ID，message role 限制为 `user`、`assistant`、`tool`，tool_calls 为 JSON 可空列。
- Produces幂等 seed 函数 `seed_faq(session)`，插入退货政策、配送时效、支付方式、退款处理等 FAQ；退货文案含完整关键词“退货政策是什么”。

- [ ] **Step 1: 写数据库专用配置、四表和 seed 的失败测试**

```python
def test_seed_can_read_database_url_without_model_credentials():
    assert Settings.database_url_from_env({"DATABASE_URL": "sqlite:///seed.db"}) == "sqlite:///seed.db"


def test_seed_is_idempotent_and_models_store_tool_calls(db_session):
    seed_faq(db_session)
    seed_faq(db_session)
    assert db_session.scalar(select(func.count()).select_from(FAQ)) == 4
    db_session.add(Conversation(conversation_id="demo-1", user_id="demo-user", status="open"))
    db_session.flush()
    message = Message(conversation_id="demo-1", role="assistant", content="", tool_calls=[{"id": "call-1"}])
    db_session.add(message)
    db_session.commit()
    saved = db_session.get(Message, message.id)
    assert saved.tool_calls[0]["id"] == "call-1"
    assert saved.conversation.user_id == "demo-user"


def test_ticket_is_linked_to_conversation(db_session):
    db_session.add(Conversation(conversation_id="ticket-conv", user_id="demo-user", status="open"))
    db_session.flush()
    db_session.add(Ticket(ticket_no="T1001", conversation_id="ticket-conv", description="商品故障", ticket_type="退货", status="open"))
    db_session.commit()
    assert db_session.get(Ticket, "T1001").conversation.conversation_id == "ticket-conv"


def test_mysql_faq_ddl_avoids_unique_index_on_text_question():
    from sqlalchemy.dialects.mysql import dialect
    from sqlalchemy.schema import CreateTable

    ddl = str(CreateTable(FAQ.__table__).compile(dialect=dialect()))
    assert "UNIQUE (question)" not in ddl
```

- [ ] **Step 2: 确认测试红灯**

Run: `python3 -m pytest tests/test_models.py::test_seed_is_idempotent_and_models_store_tool_calls -q`
Expected: FAIL，因为 ORM 和 seed 尚不存在。

- [ ] **Step 3: 实现数据库 URL 读取、四个 ORM 模型、SQLite test fixture 和 seed**

`database_url_from_env()` 只解析项目 `.env` 和环境映射里的数据库 URL，不验证 MODEL/API_KEY/BASE_URL；seed CLI 因此无需配置模型。FAQ 以 `(question, answer, category)` 为字段；Conversation 有 `user_id`、状态、创建时间和 messages 关系；Message 有会话外键、内容、JSON tool_calls、tool_call_id 和时间；Ticket 有工单号主键、会话外键、问题描述、类型、状态、创建时间。`create_tables(engine)` 调用 ORM metadata 建立四表。seed 用问题文本查询后再新增，使用能被“退货政策是什么”按 LIKE 命中的问题文案，且重复运行不重复插入。

- [ ] **Step 4: 增加 MySQL Compose 并验证模型**

Compose 使用 MySQL 8，设置健康检查和本地演示库参数。运行 `python3 -m pytest tests/test_models.py -q`、`docker compose config`。若 Docker daemon 可用，运行 `docker compose up -d db` 后执行 seed 并查询四表。

- [ ] **Step 5: 提交阶段产物**

```bash
git add app/db scripts/seed.py docker-compose.yml tests/conftest.py tests/test_models.py app/config.py tests/test_config.py pyproject.toml dev-notes/ch02.md docs/superpowers/plans/2026-09-23-ch02-function-calling.md
git commit -m "feat: add mysql persistence models and faq seed"
```

## Task 3: 五个业务工具和注册/执行基础设施

**Files:**
- Create: `app/tools/__init__.py`, `app/tools/business.py`, `app/tools/registry.py`, `tests/test_tools.py`
- Modify: `app/config.py`, `tests/conftest.py`, `dev-notes/ch02.md`

**Interfaces:**
- Consumes ORM Session factory and `Settings.tool_timeout_seconds`、`Settings.tool_max_retries`。
- Produces `build_tools(session_factory, conversation_id) -> list[BaseTool]`，恰好注册五个规定工具；`ToolRegistry(tools)`；`ToolRunner(registry, timeout_seconds, max_retries).run(name, args, tool_call_id) -> ToolResult`。
- ToolResult 字段：`tool_name`、`tool_call_id`（由编排层关联）、`content`、`is_error`。

- [ ] **Step 1: 写 FAQ 命中/漏召回和 runner 校验失败测试**

```python
def test_registry_contains_only_five_business_tools(tools):
    assert {tool.name for tool in tools} == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}


def test_demo_data_tools_return_labeled_sample_data(tools_by_name):
    assert "模拟" in tools_by_name["query_order"].invoke({"order_id": "1001"})
    assert "模拟" in tools_by_name["query_product"].invoke({"product_query": "耳机"})
    assert "模拟" in tools_by_name["query_logistics"].invoke({"order_id": "1001"})


def test_create_ticket_uses_bound_conversation(db_session, tools_by_name):
    result = tools_by_name["create_ticket"].invoke({"description": "商品故障", "ticket_type": "退货"})
    ticket = db_session.scalar(select(Ticket).where(Ticket.ticket_no == parse_ticket_no(result)))
    assert ticket.conversation_id == "demo-tools"


def test_query_faq_uses_literal_like_and_misses_postage(db_session, tools):
    registry = ToolRegistry(tools)
    assert "命中" in registry.get("query_faq").invoke({"query": "退货政策是什么"})
    assert "未命中" in registry.get("query_faq").invoke({"query": "邮费是多少"})


def test_invalid_tool_arguments_are_rejected_without_retry(runner, runner_attempt_counter):
    with pytest.raises(ToolInputError):
        runner.run("query_order", {"unexpected": True}, "call-invalid")
    assert runner_attempt_counter["query_order"] == 0


def test_runner_rejects_unknown_tool_without_execution(runner):
    with pytest.raises(UnknownToolError):
        runner.run("drop_database", {}, "call-x")


def test_runner_retries_timeout_and_returns_safe_error(runner_with_flaky_and_slow_tools):
    transient, always_fail, slow = runner_with_flaky_and_slow_tools
    assert transient.run("flaky", {}, "call-1").is_error is False
    assert transient.attempts == 2
    exhausted = always_fail.run("always_fail", {}, "call-exhausted")
    assert exhausted.is_error is True
    assert always_fail.attempts == 1 + always_fail.max_retries
    timed_out = slow.run("slow", {}, "call-2")
    assert timed_out.is_error is True
    assert "Traceback" not in timed_out.content
```

- [ ] **Step 2: 确认测试红灯**

Run: `python3 -m pytest tests/test_tools.py -q`
Expected: FAIL，因为工具和注册表尚未实现。

- [ ] **Step 3: 实现五个 `@tool` 与显式注册表**

实现 `tools_by_name` 和 `runner_attempt_counter` 测试夹具；前者绑定会话 ID `demo-tools`，后者记录工具函数实际执行次数，以便证明 Schema 校验错误发生在函数执行前。FAQ 用 SQLAlchemy 参数绑定表达式 `FAQ.question.like(f"%{query}%")`。演示订单、商品、物流各自返回含“模拟”标记的随机字段，输入分别为 `order_id`、`product_query`、`order_id`。create_ticket 闭包固定当前 conversation_id，仅接受 description 和 ticket_type，并返回带有 `ticket_no` 的 JSON。工具函数的类型标注定义模型可见输入 Schema。

- [ ] **Step 4: 实现执行错误、超时与有限重试并验证**

Runner 执行前按注册表检查工具名，再以生成的 Pydantic Schema 校验 args。参数/未知工具错误不重试；临时异常按配置次数重试；超时和最终异常转换为不含堆栈的结构化 `ToolResult(is_error=True)`。对 Step 1 的失败测试最小实现，并补齐 Step 1 中 runner 执行次数恰为 `max_retries + 1`、无效 Schema 不重试的断言。

Run: `python3 -m pytest tests/test_tools.py -q`
Expected: 全部 PASS；重试耗尽时执行次数严格为 `max_retries + 1`。

- [ ] **Step 5: 提交阶段产物**

```bash
git add app/tools app/config.py tests/test_tools.py tests/conftest.py dev-notes/ch02.md
git commit -m "feat: add registered business tools"
```

## Task 4: 单轮聊天编排、数据库消息流水与工具回灌

**Files:**
- Create: `app/prompts.py`, `app/services/__init__.py`, `app/services/chat.py`, `tests/test_chat.py`
- Modify: `app/schemas.py`, `app/tools/registry.py`, `tests/conftest.py`, `dev-notes/ch02.md`

**Interfaces:**
- Produces `ChatRequest(conversation_id: str, message: str, user_id: str = "demo-user")`。
- Produces `ChatService(session_factory, model_factory, tool_runner_factory).stream_events(request) -> Iterator[dict]`；事件为 tool_status、token、done、error。默认 model factory 用配置构造 `ChatOpenAI`，决策阶段调用 `model.bind_tools(tools).stream(messages)`，回灌阶段调用未绑定工具的 `model.stream(messages)`。
- 初始绑定工具的流只允许 0 或 1 个 tool call；一个调用时 tool message 用同一 tool_call_id；回灌后的第二次模型流不绑定工具。

- [ ] **Step 1: 写完整工具轮次与普通聊天的失败测试**

```python
def test_tool_round_persists_request_result_and_streams_final_tokens(chat_service, db_session):
    events = list(chat_service.stream_events(ChatRequest(conversation_id="demo-1", message="订单 1001 的物流到哪了")))
    assert [event["event"] for event in events] == ["tool_status", "token", "token", "done"]
    messages = db_session.scalars(select(Message).order_by(Message.id)).all()
    assert [message.role for message in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1].tool_calls[0]["id"] == messages[2].tool_call_id


def test_multiple_tool_calls_are_rejected_without_execution(chat_service, tool_runner):
    events = list(chat_service.stream_events(ChatRequest(conversation_id="demo-2", message="查订单和商品")))
    assert events[-1]["event"] == "error"
    assert tool_runner.calls == []


def test_no_tool_answer_streams_and_persists(chat_service, db_session):
    events = list(chat_service.stream_events(ChatRequest(conversation_id="demo-3", message="你好")))
    assert [event["event"] for event in events] == ["token", "token", "done"]
    assert db_session.scalar(select(Message).where(Message.role == "assistant")).content == "你好。"


def test_tool_error_is_replayed_as_matching_tool_message(chat_service, tool_runner, fake_model):
    events = list(chat_service.stream_events(ChatRequest(conversation_id="demo-4", message="查物流")))
    assert events[-1]["event"] == "done"
    assert fake_model.second_call_messages[-1].tool_call_id == tool_runner.last_result.tool_call_id
    assert tool_runner.last_result.is_error is True


def test_model_failure_emits_safe_error_and_keeps_user_message(chat_service, db_session):
    events = list(chat_service.stream_events(ChatRequest(conversation_id="demo-5", message="查商品")))
    assert events[-1] == {"event": "error", "data": {"message": "暂时无法处理，请稍后再试。"}}
    assert db_session.scalar(select(Message).where(Message.role == "user")).content == "查商品"
```

- [ ] **Step 2: 确认测试红灯**

Run: `python3 -m pytest tests/test_chat.py -q`
Expected: FAIL，因为 ChatService 尚不存在。

- [ ] **Step 3: 实现数据库持久化顺序和一次工具决策**

先提交 conversation/user message；初始模型流不持有数据库事务。无工具时保存 assistant 文本。单一工具申请时先持久化 AIMessage 的原始 tool_calls，再发 tool_status、执行一次并持久化 tool role 结果，最后构造带匹配 tool_call_id 的 ToolMessage。

- [ ] **Step 4: 回灌模型、流式保存并验证错误分支**

将 system、从数据库读取的会话消息、本轮 assistant tool-call 和 ToolMessage 交给未绑定工具的流式模型；每个文本增量产生 token 事件，完整回答聚合保存后才产生 done。工具错误作为 ToolMessage 回灌；模型/数据库错误产生通用 error。实现 Step 1 的无工具、工具错误和模型失败测试。

Run: `python3 -m pytest tests/test_chat.py -q`
Expected: 普通回答与工具最终回答保留 token 事件；工具模型调用最多两次（工具决策一次、结果收敛一次），工具 runner 每轮最多执行一次。

- [ ] **Step 5: 提交阶段产物**

```bash
git add app/prompts.py app/services app/schemas.py app/tools/registry.py tests/test_chat.py tests/conftest.py dev-notes/ch02.md
git commit -m "feat: orchestrate persisted single tool chat"
```

## Task 5: SSE API 与协议测试

**Files:**
- Modify: `app/main.py`, `app/schemas.py`
- Create: `tests/test_api.py`
- Modify: `dev-notes/ch02.md`

**Interfaces:**
- `POST /api/v1/chat/stream` 接收 `ChatRequest`，响应 `text/event-stream`。
- SSE 格式为 `event: <type>\ndata: <JSON>\n\n`；事件类型为 `tool_status`、`token`、`done`、`error`。

- [ ] **Step 1: 写 SSE 顺序和错误帧测试**

```python
def test_chat_stream_returns_tool_status_tokens_and_done(client):
    response = client.post("/api/v1/chat/stream", json={"conversation_id": "api-1", "message": "查退货政策"})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert parse_sse(response.text) == ["tool_status", "token", "done"]


def test_chat_stream_turns_service_failure_into_error_event(client_with_failing_service):
    response = client_with_failing_service.post("/api/v1/chat/stream", json={"conversation_id": "api-2", "message": "查商品"})
    assert parse_sse(response.text)[-1] == "error"
    assert "Traceback" not in response.text
```

- [ ] **Step 2: 确认测试红灯**

Run: `python3 -m pytest tests/test_api.py -q`
Expected: FAIL，因为当前 API 尚无聊天路由。

- [ ] **Step 3: 注入 ChatService 并编码 SSE**

创建 `create_app(chat_service=None)` 供测试注入；路由迭代服务事件并按 SSE wire format 编码 JSON data，设置 `Cache-Control: no-cache`。流开始后捕获异常并发送通用 error 帧。

- [ ] **Step 4: 验证 API 成功和失败响应**

实现 Step 1 的错误帧测试；断言完整 event/data 顺序和 done 中的 conversation_id。运行 `python3 -m pytest tests/test_api.py -q` 和 `python3 -m pytest -q`。
Expected: API 与此前各层测试全绿。

- [ ] **Step 5: 提交阶段产物**

```bash
git add app/main.py app/schemas.py tests/test_api.py dev-notes/ch02.md
git commit -m "feat: expose tool chat through sse"
```

## Task 6: Vibe Coding 聊天页工具轨迹

**Files:**
- Create: `app/static/index.html`
- Modify: `app/main.py`, `dev-notes/ch02.md`

**Interfaces:**
- `GET /` 返回聊天页。
- 浏览器 POST `conversation_id`、`user_id`、`message` 到 `/api/v1/chat/stream` 并消费上述 SSE 协议。

- [ ] **Step 1: 直接实现聊天交互与徽章**

以简洁客服聊天布局实现输入框、发送按钮、用户/助手气泡；每轮助手气泡预留工具轨迹。收到 tool_status 显示对应工具名和“查询中”，收到 token 按序追加文字，收到 done 移除状态并保留工具徽章；error 展示安全的通用提示。使用 `fetch` 的 ReadableStream 解析 POST SSE，不依赖原生 EventSource（其不支持本请求体）。对历史消息保留 `conversation_id` 并由后端查库续聊。

- [ ] **Step 2: 浏览器手工验收**

运行 API 后打开 `/`：发送“订单 1001 的物流到哪了”，确认徽章和物流回答；发送“退货政策是什么”，确认 FAQ 工具徽章和答案；刷新后复用会话 ID 验证历史记录仍存在。聊天页任务按用户指定的 Vibe Coding 方式直接改，不为此 UI 单独编写测试或发起单独 code review。

- [ ] **Step 3: 提交阶段产物**

```bash
git add app/static/index.html app/main.py dev-notes/ch02.md
git commit -m "feat: show tool trace in support chat"
```

## Task 7: 评估样例、运行手册和端到端验收

**Files:**
- Create: `tests/fixtures/faq_cases.json`, `README.md`
- Modify: `scripts/seed.py`, `docker-compose.yml`, `tests/test_tools.py`, `dev-notes/ch02.md`

**Interfaces:**
- 标注样例包含 `退货政策是什么 => hit` 和 `邮费是多少 => expected_miss`。
- README 命令覆盖依赖安装、MySQL 启动、seed、应用启动、pytest 和浏览器演示。

- [ ] **Step 1: 写 FAQ 标注评估集并验证预期结果**

```json
[
  {"query": "退货政策是什么", "expected": "hit"},
  {"query": "邮费是多少", "expected": "expected_miss"}
]
```

先在 `tests/test_tools.py` 写 `test_faq_evaluation_cases_match_labels`：逐条读取 JSON，用 FAQ 工具实际执行 query，断言 hit 样例有结果、expected_miss 样例无结果；运行并确认漏召回按预期通过后，再提交标注 JSON。

- [ ] **Step 2: 运行完整自动化验证**

Run: `python3 -m pytest -q`
Expected: 所有测试通过；报告真实测试计数，不把跳过的真实模型调用算作通过。

- [ ] **Step 3: 写演示命令并验证 Docker 配置**

README 写出：`python3 -m pip install -e '.[dev]'`、`docker compose up -d db`、`python3 -m scripts.seed`、`uvicorn app.main:app --reload`、`python3 -m pytest -q`。运行 `docker compose config`；在 Docker daemon 可用时重建空数据库、执行 seed 两次，确认 FAQ 行数不变。

- [ ] **Step 4: 浏览器/模型手工验收并记漏召回**

执行 spec 中的订单物流和退货政策演示；确认 FAQ 漏召回评估结果是 expected_miss。若模型凭据可用运行真实聊天，否则记录确定性 fake model 覆盖了自动化编排、真实模型手工验收未运行。`dev-notes/ch02.md` 记录命令、结果、FAQ 漏召回以及任何环境阻塞，不记录密钥或完整敏感消息。

- [ ] **Step 5: 提交收尾文档**

```bash
git add tests/fixtures/faq_cases.json README.md scripts/seed.py docker-compose.yml dev-notes/ch02.md
git commit -m "docs: add function calling demo and faq evaluation"
```
