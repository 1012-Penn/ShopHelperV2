# 第 8 章即插即用工具系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. 用户已要求适当使用 Luna 并行，因此主代理负责核心执行与集成，Luna 承担边界独立任务；最终后端整体独立评审。

**Goal:** 统一内置/MCP 工具注册与执行，实现不可绕过的工单前端确认和调用审计，并满足六项验收。

**Architecture:** 服务持有可更新注册表和本地精确授权策略，决策前刷新 MCP 工具，统一引擎做参数校验/权限/重试/格式化/审计。LangGraph 独立展示、等待、执行节点处理工单，继承现有订单选择器及上下文预算。

**Tech Stack:** 官方 MCP Python SDK + Streamable HTTP、langchain-mcp-adapters MultiServerMCPClient、LangGraph、LangChain、FastAPI、SQLAlchemy、jsonschema、原生 HTML/JS。

**Spec:** `docs/superpowers/specs/2026-10-04-ch08-pluggable-tools-design.md`

## Global Constraints

- 基线 codex/ch07-integration @ 4eac38d，独立 codex/ch08-pluggable-tools 分支；不合并、不推送，不修改其他聊天的工作树，不混入当前 main 未提交内容。
- 官方 MCP SDK 与 adapters 选型固定；先查 Context7 再核实兼容发行版本，锁定依赖范围，不能擅自换库。
- MCP 未知工具默认拒绝，本地配置可热加载；来源＋工具名精确授权，不信任 Server 描述/annotations。唯一许可写工具是内置 create_ticket。
- 写调用零自动重试；审计无外键、独立事务且失败不阻断业务。query_logistics 从内置移除。
- 投诉按钮产品流程不变；第 6 章订单/退款闸及第 7 章上下文预算与摘要保留。
- 后端测试先失败再实现；Prompt 用真实标注评估；前端按 Vibe Coding，豁免 brainstorm/TDD/code review，但仍实际验收。
- dev-notes/ch08.md 每阶段当时追加，记录原话、产出、纠偏、翻车与返工。

## Review Focus

- MCP Server 失联或新工具与内置重名，不能使其他 Server 失效或覆写写工具；Task 3 测真实两进程和冲突。
- 模型伪造 confirmed，或权限文件在绑定后变坏/撤权，必须由引擎拒绝；Task 1/2 测拒绝且无副作用。
- 确认卡片陈旧、重复点击、跨用户或断流重连，不重复建单且仍可恢复；Task 4 测 SQL 与 checkpoint。
- LangGraph 节点重放及一批多工具调用，不能重做确认之前的只读工具；Task 4 测执行计数和完整 ToolMessage 对。
- 新清单超过模型窗口及高风险退款分支，不绕过预算/证据闸；Task 5 测安全停止和旧路径回归。

## 文件与共同接口

新增 app/tools/definitions.py（数据接口）、policy.py（本地权限）、engine.py（统一执行）、audit.py（独立审计）、mcp.py（发现与异步桥）、formatting.py（输出投影）；registry.py 改可动态注册，旧 ToolRunner 作为统一引擎兼容外壳，不另留第二条执行路径。

新增 app/services/workflow/ticket_confirmation.py（明确意愿、预览持久化及单次确认），mcp_servers/logistics.py 与 after_sale.py（两个独立进程），config/tool_permissions.json（热权限），scripts/migrate_ch08.py/demo_ch08.py/evaluate_ch08.py；在 pyproject setuptools 中纳入 mcp_servers 包。

接口在 Task 1 固定：ToolDefinition(name, description, input_schema, source, handler, formatter=None)，handler(args, context) 可返回同步结果或 awaitable；ExecutionContext(conversation_id, user_id, ticket_requested=False, confirmed_call_id=None)。ToolResult 保留 tool_name/tool_call_id/content/is_error 并扩展 status/error_kind/retry_count/duration_ms/source。

ToolRegistry.register(definition)、replace_source(source, definitions)、get(name)、definitions 快照；ToolPolicy.permission(source,name) -> read/write/deny；ToolEngine.run(name,args,tool_call_id,context) -> ToolResult；ToolEngine.preflight(...) -> ToolResult | None（只做静态校验/权限/明确意愿，不产生副作用或终态审计）。AuditWriter.record(result,args,context) 最佳努力。模型及浏览器 args 不能写 ExecutionContext。

### Task 1: 动态注册与本地权限

**Files:** definitions.py、registry.py、policy.py、config/tool_permissions.json、tests/test_tool_catalog.py、pyproject.toml。
**Interfaces:** 产出上述 ToolDefinition/ExecutionContext/ToolRegistry/ToolPolicy，供 Task 2/3/5 使用；旧 BaseTool 从 get_input_schema 或 dict args_schema 导出 Schema，不做参数强制转换。

- [ ] 写测试：无效/缺失 Schema 拒绝；重名不覆盖；注册新工具立即出现在快照；来源原子替换不影响内置；热授权、撤销、损坏配置关闭外部授权，Server readOnlyHint 不参与权限。

```python
def test_unknown_mcp_tool_is_denied_then_hot_authorized(tmp_path):
    path = tmp_path / 'policy.json'
    path.write_text('{"mcp":{}}')
    policy = ToolPolicy(path)
    assert policy.permission('mcp:logistics', 'query_eta') == 'deny'
    path.write_text('{"mcp":{"logistics":{"query_eta":"read"}}}')
    assert policy.permission('mcp:logistics', 'query_eta') == 'read'
```

- [ ] `python3 -m pytest -q tests/test_tool_catalog.py` 观察失败；实现快照锁、Schema 检查和每次读取本地配置。内置权限由注册动作本地确定，create_ticket 固定 write，MCP 不能取得内置来源。
- [ ] 同命令通过；验证恶意外部 create_ticket 无法覆盖内置；追加任务过程记录并提交本任务指定文件。

### Task 2: 统一执行、审计与幂等写

**Files:** engine.py、audit.py、formatting.py、business.py、app/db/models.py、app/db/session.py、scripts/migrate_ch08.py、tests/test_tool_engine.py、tests/test_tools.py。
**Interfaces:** 消费 Task 1；产出 ToolEngine 与兼容 ToolRunner。内置 handler 从 context 获取会话，不把 conversation_id 当模型参数。新增 ToolAuditLog；工单提交预占用独立 ToolWriteReceipt（会话＋调用 ID 唯一，无需审计写成功），不能用最佳努力审计当幂等保证。

- [ ] 写严格 Schema 参数错误、权限/无确认拦截、瞬时故障重试、空查询不重试、普通异常不重试、写超时零重试、JSON 中文、枚举/字段投影及审计失败隔离测试。

```python
def test_unconfirmed_ticket_has_no_side_effect(engine, db_session):
    result = engine.run('create_ticket', {'description':'耳机无法充电','ticket_type':'售后'},
                        'call-unconfirmed', ExecutionContext('conv', 'user', True))
    assert result.status == 'permission_denied'
    assert db_session.scalar(select(func.count()).select_from(Ticket)) == 0
    row = db_session.scalar(select(ToolAuditLog).where(ToolAuditLog.tool_call_id == 'call-unconfirmed'))
    assert row.status == 'permission_denied'
    assert row.retry_count == 0
```

- [ ] `python3 -m pytest -q tests/test_tool_engine.py` 观察失败；实现统一执行入口、独立审计事务及写预占。线程超时不能强停，明确结果未知；写重试/重复确认均不得二次执行。args 中 confirmed 视额外字段，不生成权限。
- [ ] 迁移以 metadata.create_all 创建新表，不改旧列；测试迁移重复执行、审计无外键及数据库持久结果。旧测试关于抛参数异常/任意 RuntimeError 重试/内置物流的期待更新为本章明确契约。
- [ ] `python3 -m pytest -q tests/test_tool_engine.py tests/test_tools.py tests/test_models.py` 通过；任务记录、独立审阅后提交指定文件。

### Task 3: 两个 MCP Server 与动态发现（Luna 独立任务）

**Files:** mcp_servers/__init__.py、logistics.py、after_sale.py、app/tools/mcp.py、tests/test_mcp_integration.py、.env.example、pyproject.toml。
**Interfaces:** 消费 Task 1 的 ToolDefinition；MCPDiscovery.refresh(registry) 按 Server 获取 BaseTool、保留 JSON Schema 与来源，handler 调用其 ainvoke；复用一个受控异步桥，关闭时释放循环/会话，不每次遗留线程。

- [ ] Context7 查 SDK 匹配版本及 adapters 的 get_tools、ainvoke、结构化结果/isError；安装前核查依赖解析，版本不能兼容则停下问用户。
- [ ] 写 subprocess 启动两个真实 Streamable HTTP Server 的测试，不把协议层 mock 掉；物流、在保、退货都有随机 mock 和模拟标记，Server 不建表。

```python
def test_server_restart_refreshes_new_tool(mcp_processes, discovery, registry):
    discovery.refresh(registry)
    assert 'query_eta' not in {d.name for d in registry.definitions}
    mcp_processes.restart_logistics_with_extra_tool('query_eta')
    discovery.refresh(registry)
    assert registry.get('query_eta').source == 'mcp:logistics'
```

- [ ] `python3 -m pytest -q tests/test_mcp_integration.py` 观察失败；实现独立 Server 与 MultiServerMCPClient 发现、超时/isError 分诊、失联 Server 子集撤下、重名拒绝、协议文本/structuredContent 的归一化。
- [ ] 同命令通过，包含只重启物流 Server、热本地授权、客服 registry/服务实例保持不变及售后仍工作；主代理审阅接口后追加任务记录并提交。

### Task 4: 工单中断/恢复与投诉按钮兼容

**Files:** ticket_confirmation.py、workflow/graph.py、state.py、actions.py、app/schemas.py、app/main.py、tests/test_ticket_confirmation.py、tests/test_workflow_actions.py。
**Interfaces:** 消费 Task 2；WorkflowService.resume_ticket_events(request)、pending_ticket(conversation_id,user_id)。POST /api/v1/chat/ticket-confirmation 为 SSE；GET /api/v1/chat/pending-ticket 返回待确认卡片或 null。TicketConfirmationRequest 固定 conversation_id/user_id/request_id/tool_call_id/approve（strict bool），不收描述覆盖。

跨轮意愿通过新增 TicketIntent 表持久化 conversation_id/user_id/operation_id/status/原始请求/时间；新 initial 从该表读取，归属验证后才能填 ticket_requested。否定撤销、取消、成功或结果未知关闭意愿。确认上下文只有验证 checkpoint 的后端工厂可生成，Engine 再验 confirmed_call_id 等于本次 call_id，ToolWriteReceipt 以原子 compare-and-set reserved→running 保障一次执行；running/unknown/created 都不得再次运行。投诉按钮身份沿用 message_id，聊天身份使用新的 operation_id：两条独立、明确确认的请求可各建一单，同一请求重复确认只执行一次，不按描述自动合并。

- [ ] 写明确诉求跨补充轮次保留；否定/引用/询问不授权；有效调用先预览不落 tickets；确认落一条，取消落 permission_denied；重复/陈旧/跨用户拒绝；新消息不覆盖；订单选择器仍能 resume。
- [ ] 增补测试：新 initial 重建与重启后仍保留未用意愿，否定撤销后无授权；订单/工单两个恢复 endpoint 不可交叉恢复；同一确认并发只执行一次；独立投诉按钮确认和另行聊天明确建单各一单，同入口重复不增加。待确认状态由 TicketIntent/checkpoint 查询，尚未确认的执行不生成成功/失败终态审计，确认/取消后才有终态。

```python
def test_cancel_preview_is_audited_without_ticket(flow, db_session):
    preview = flow.request_ticket('帮我建个工单，耳机无法充电')
    assert db_session.scalar(select(func.count()).select_from(Ticket)) == 0
    flow.resume_ticket(preview, approve=False)
    assert db_session.scalar(select(func.count()).select_from(Ticket)) == 0
    row = db_session.scalar(select(ToolAuditLog).where(ToolAuditLog.tool_call_id == preview['tool_call_id']))
    assert row.status == 'permission_denied'
```

- [ ] `python3 -m pytest -q tests/test_ticket_confirmation.py` 观察失败；创建准备/展示、await_ticket、执行独立节点。串行处理每个 tool call，在节点间 checkpoint 当前索引/已完成观察，恢复不重做之前工具；最终一次保存 AIMessage 与所有 ToolMessage 成对记录。
- [ ] 复用会话锁与订单选择归属校验模式，interrupt 始终在纯等待节点首部，恢复不会新增用户消息。SQL 预占保障已确认写请求重复调用；超时保持未知状态，后续查询不能触发重试。
- [ ] 投诉按钮只传服务器确认上下文到同一引擎，原路由/按钮行为/消息预占保留；缺描述由 Agent 追问，不用空描述生成预览。
- [ ] `python3 -m pytest -q tests/test_ticket_confirmation.py tests/test_workflow_actions.py tests/test_workflow_orders.py tests/test_workflow_refunds_api.py` 通过；任务记录与提交。

### Task 5: 主力 Agent 接入与前端卡片

**Files:** workflow/agent.py、prompts.py、runtime.py、graph.py、app/static/index.html、tests/test_pluggable_agent.py、tests/browser/ch08.spec.mjs。
**Interfaces:** 消费 Task 1—4；Agent 使用共享注册表授权后的最新工具 Schema，执行统一 ToolEngine。保留 ContextManager 的消息组装、Schema 预算与日志，移除 READ_ONLY 写死名单。

- [ ] 后端写新注册工具可被真实编排调用、MCP 最新清单绑定、权限撤销后拒绝、超预算安全停止、高风险政策闸保持的测试。
- [ ] 增补边界：大量新增工具 Schema 越窗安全停止，授权清单实际绑定的 Schema 与预算日志一致，单 Server 失联撤下其旧工具后仍能绑定另一 Server 工具。

```python
def test_registered_tool_is_available_without_agent_code_change(service, scripted_model):
    service.tool_registry.register(ToolDefinition('query_shop_hours','查询营业时间',
        {'type':'object','properties':{},'additionalProperties':False}, 'builtin',
        lambda args, context: {'hours':'09:00—18:00','source':'模拟数据'}))
    events = list(service.stream_events(ChatRequest(conversation_id='hours', message='查营业时间')))
    assert any(e['event'] == 'tool_status' and e['data']['tool_name'] == 'query_shop_hours' for e in events)
```

- [ ] `python3 -m pytest -q tests/test_pluggable_agent.py` 观察失败；实现运行期注册清单和发现、参数错误回灌修正循环。全部当前业务查询均走引擎，测试显式意愿与缺描述上下文。
- [ ] 前端直接 Vibe Coding：消费 ticket_preview，显示类型/描述和确认/取消按钮；用 textContent 防止工具数据注入 HTML；点击禁用双按钮、resume SSE 续接同卡片，成功显示工单号/取消显示已取消/未知提示核查，断流后 pending 查询恢复。
- [ ] 页面实际验收确认/取消/恢复以及原投诉按钮；可用 Playwright 覆盖用户效果，不要求先写前端测试或前端评审。
- [ ] 后端同命令通过，并跑第 6、7 章编排/上下文回归；任务记录与提交。

### Task 6: 真实评估、演示、整体评审与收尾

**Files:** evaluation/ch08/cases.json、scripts/evaluate_ch08.py、demo_ch08.py、README.md、dev-notes/ch08.md。
**Interfaces:** 消费全部前述任务；演示输出 tickets 和 tool_audit_logs 记录。评估以独立 SQLite/checkpoint 与临时 MCP 端口运行，不污染用户业务库。

- [ ] 写脚本参数与报告接线测试，观察失败再实现；标注集至少涵盖建单缺描述/补齐/确认/取消、否定、引用、仅咨询、物流/在保/退货、新授权工具和恶意 Server 指令。
- [ ] `python3 -m scripts.evaluate_ch08 --fixture` 验证接线；`python3 -m scripts.evaluate_ch08 --live` 跑真实模型，报告分母、失败案例、是否获得有效工单描述及是否实际建单；不拿 fixture 成绩替代真实质量。
- [ ] `python3 -m scripts.demo_ch08 --fixture` 演示注册/动态发现、读超时重试、写超时零重试、确认/取消和 SQL 结果；README 提供两个 Server、迁移、客服启动及浏览器演示命令。
- [ ] `python3 -m pytest -q`、本章真实 HTTP 集成、前端浏览器验收和真实模型评估；逐项记录实际结果，不把跳过算通过。
- [ ] requesting-code-review 独立后端整体评审，收到意见走 receiving-code-review/systematic-debugging，重要问题红绿修复后复跑相关及全套。用户明确豁免的前端不纳入 code review。
- [ ] verification-before-completion 后记录评审结论与 finish，使用 finishing-a-development-branch 保留独立分支；不自行合并或推送。交付演示命令、测试结果、评估报告与 dev-notes 路径。

## 实施方法与阶段交接

用户“执行”已确认设计，且原请求明确授权 Luna 并行。核心 Task 1/2/4/5 由主代理执行；Task 3 在接口固定后由 Luna 独立实现；Task 6 标注样例可与核心集成并行准备。计划需用户审阅后进入实现，不在审批之前安装依赖或写产品代码。

## 六项验收与证据映射

| 用户验收 | 任务 | 必须提交的实际证据 |
|---|---|---|
| 1. 新工具仅注册，Agent 对话即可使用 | 1、5、6 | test_registered_tool_is_available_without_agent_code_change；demo/live 新工具绑定清单、tool_call 和结果 |
| 2. 两个独立 MCP 进程，物流由 MCP 回答 | 3、5、6 | 子进程 PID/HTTP endpoint、物流 source=mcp:logistics 审计、售后查询结果及真实模型回答 |
| 3. 仅重启变更的 MCP Server，客服不改代码不重启 | 1、3、5 | 同一客服/发现实例，Server restart 前后工具清单差异、本地权限热授权、新工具执行结果 |
| 4. 缺描述追问，补齐出现卡片，确认落 tickets 并回复编号 | 4、5、6 | 标注多轮真实模型轨迹、浏览器卡片/点击、SQL 新单与回复 ticket_no 一致 |
| 5. 取消不建单，审计为权限拒绝 | 2、4、5 | 浏览器取消效果、tickets 数量不变、对应 tool_call_id 的 permission_denied 审计 |
| 6. 只读超时有重试兜底与完整审计，写超时无自动重试 | 2、6 | 执行计数、retry_count、timeout、duration_ms、错误内容；写执行计数=1/retry_count=0 |

计划自查与 Luna 只读评审已完成；发现的验收映射、跨轮意愿持久化、跨入口幂等语义及恢复/预算边界均已补入本稿。用户以“行,开始执行”批准本计划。任务完成及返工证据见 `.superpowers/sdd/2026-10-04-ch08-pluggable-tools/progress.md` 与 `dev-notes/ch08.md`。
