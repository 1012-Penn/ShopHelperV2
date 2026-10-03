# ch05 Workflow Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. 用户已明确授权“直接生成计划并且实行”，不重复请求执行批准；Luna 协助独立前端与评审任务。

**Goal:** 交付固定四路 Workflow、受控 ReAct Agent、持久 State 和用户独立选择的两个操作按钮。

**Architecture:** 先手写普通 Python Agent 循环，再将决策、工具、最终流式生成拆入一张 StateGraph。SQL 记录业务会话与按钮建议，官方 SQLite checkpointer 记录图 State；工单仅通过已确认的按钮接口调用原有工具。

**Tech Stack:** Python 3.10+、LangGraph 1.2.x、LangChain 1.4.x、FastAPI、SQLAlchemy 2、官方 langgraph-checkpoint-sqlite、原生 HTML/CSS/JavaScript。

**Spec:** `docs/superpowers/specs/2026-10-03-ch05-workflow-agent-design.md`

## Global Constraints

- 四出口：knowledge（商品咨询、退款退货）、business（物流、订单、售后）、complaint、chitchat。分类失败是错误路径。
- 常见闲聊整个请求零调用；其他闲聊只分类一次；投诉不进 Agent。
- 知识强制先检索后闸门；business 不检索、不走闸门。重排阈值 0.05 基线沿用 ch04，dense 阈值 0.60 沿用 ch03。
- Agent 仅绑定既有 query_order/query_product/query_logistics，不绑定 query_faq/create_ticket。
- 4 次决策、6 次业务工具执行、每调用最多 512 输出、当轮模型总预算 12000（检索内部既有归一化调用另列）。usage 缺失保守估算，不能称精准计费。
- 官方 SQLite checkpointer、单进程演示，不自作主张换后端或承诺多副本。消息与 checkpoint 的两个持久系统不承诺跨库原子事务。
- 转人工仅前端模拟；工单状态先 submitting 再调用原工具，created 重复返回原单号，submitting/unknown 禁止自动重试。
- 涉及库用法先查 Context7；纯 prompt 用标注集 live 验证；逐阶段记录 dev-notes/ch05.md。

## Review Focus

- 用户发知识问题后立刻发物流：新一轮不沿用旧证据/按钮，任务 3/4 测字段重置。
- 模型伪造 create_ticket 工具调用或非法 JSON：不能写单、不能无期限循环，任务 4 测非法调用。
- 同一投诉按钮并发/重复请求或工具超时：至多一次调用写工具，未知不谎报成功，任务 2 测状态预占与恢复。
- 最终流中途失败、空结果或超预算：不发送成功 done、不保留未配对工具消息，任务 4/5 测错误与预算。
- 两个建议都不点或取消确认：无请求与写单，能继续普通消息；任务 6 浏览器验证。

---

### Task 1: 裸循环与共同预算策略

**Files:** Create `app/services/workflow/bare.py`, `app/services/workflow/policy.py`, `app/services/workflow/__init__.py`, `tests/test_bare_agent.py`, `scripts/demo_bare_agent.py`.

**Interfaces:** `Limits(max_decisions=4,max_tool_calls=6,max_output_tokens=512,max_tokens=12000)`；`run_bare_agent(messages, call_model, run_tool, limits=Limits()) -> dict`，普通 dict 消息与工具请求，不依赖编排框架；`BudgetExceeded` 和模型消息输入预算估算供任务 4 使用。

- [ ] 写两步循环、无工具结束、缺信息正文、非法工具数据、重复工具/步数/预算停止测试；真实业务依赖只注入 LLM/工具网络边界。
```python
def test_order_observation_drives_logistics():
    seen = []
    responses = iter([
        {'content':'','tool_calls':[{'id':'o','name':'query_order','args':{'order_id':'1001'}}]},
        {'content':'','tool_calls':[{'id':'l','name':'query_logistics','args':{'order_id':'1001'}}]},
        {'content':'物流运输中','tool_calls':[]},
    ])
    def model(messages, max_tokens):
        seen.append(list(messages))
        return next(responses)
    result = run_bare_agent([{'role':'user','content':'先查订单再查物流'}], model,
                            lambda name,args,call_id: '模拟数据')
    assert result['answer'] == '物流运输中'
    assert result['tool_calls'] == 2
    assert seen[1][-1]['tool_call_id'] == 'o'
    assert seen[2][-1]['tool_call_id'] == 'l'
```
- [ ] `python3 -m pytest tests/test_bare_agent.py -q` 观察缺实现而红。
- [ ] 实现最多调用次数/执行次数/总预算检查、工具消息配对和重复请求停止，裸循环示例运行 `python3 -m scripts.demo_bare_agent --fixture`。
- [ ] 重跑专项、记录红绿证据与 demo 输出；只提交本任务文件和阶段笔记。

### Task 2: 建议持久化、工单按钮服务与迁移

**Files:** Modify `app/db/models.py`, `app/db/session.py`; Create `app/services/workflow/storage.py`, `app/services/workflow/actions.py`, `scripts/migrate_ch05.py`, `tests/test_workflow_actions.py`.

**Interfaces:** `ConversationStore(session_factory).prepare(request)->list[BaseMessage]`, `.save_answer(conversation_id,answer,citations,actions,question)->int`；`TicketActions(session_factory,runner_factory,locks).submit(conversation_id,user_id,message_id)->dict`；`ConversationLocks.hold(conversation_id)` 用于任务 3/5。

- [ ] 写迁移重复执行保旧消息、会话归属、投诉建议不写单、点击建单/重复返回同单号、submitting/unknown不重试、未建议/跨用户拒绝测试。
```python
def test_ticket_requires_suggestion(db_session_factory):
    store = ConversationStore(db_session_factory)
    store.prepare(ChatRequest(conversation_id='c', message='我要投诉'))
    message_id = store.save_answer('c','很抱歉',[],['handoff'], '我要投诉')
    service = TicketActions(db_session_factory, runner_factory, ConversationLocks())
    with pytest.raises(ValueError):
        service.submit('c','demo-user',message_id)
    with db_session_factory() as s:
        assert s.scalar(select(func.count()).select_from(Ticket)) == 0
```
- [ ] 运行 `python3 -m pytest tests/test_workflow_actions.py -q` 看红。
- [ ] 添加 nullable Message.actions JSON，显式迁移 inspect/get_columns + ALTER TABLE，替换整个 JSON 触发 ORM 更新；提交前标记 submitting，复用原 create_ticket，禁止未知结果重试。工具执行器增加生命周期 close 并在使用处释放线程池。
- [ ] 专项与既有模型/工具测试全绿后记录、提交。

### Task 3: 固定分流、前置闸、State 和 checkpoint

**Files:** Create `app/services/workflow/state.py`, `app/services/workflow/intents.py`, `app/services/workflow/retrieval.py`, `app/services/workflow/graph.py`, `app/services/workflow/logging.py`, `tests/test_workflow_routes.py`; Modify `pyproject.toml`, `.env.example`.

**Interfaces:** `WorkflowService(session_factory,model_factory,runner_factory,retriever,checkpointer,log_path,limits=Limits(),min_score=0.05,strategy='hybrid_rerank')` exposes `.stream_events(ChatRequest)`, `.close()`；`ROUTES` 七类常量；`EvidenceAdapter.retrieve(question,category)->(snapshots,trace)` 对齐 ch03/04 检索器。

- [ ] 写七意图四路、常见闲聊无模型、投诉无 Agent、证据弱落账本、检索故障错误、分类非法、证据跨轮重置、SQLite checkpoint重开读取/会话隔离测试。
```python
def test_weak_policy_never_enters_agent(workflow):
    events = list(workflow.stream_events(ChatRequest(conversation_id='weak',message='退货政策')))
    assert events[-1]['event'] == 'done'
    state = workflow.graph.get_state({'configurable':{'thread_id':'weak'}}).values
    assert state['route'] == 'knowledge'
    assert state['gate_passed'] is False
    assert state['decisions'] == 0
```
- [ ] 运行新专项看红；Context7 再核对 add_messages/custom stream 与 SqliteSaver；安装官方 saver 包，声明兼容依赖。
- [ ] 实现 refer/classify/route/retrieve/gate/fixed/log 图节点与四路边；agent入口先接可验证的停止回复，任务 4 替换为真正循环。检索 fault 不当低置信；弱证据原话入既有账本。日志记录成功和失败路径。
- [ ] 测新专项，记录 State恢复/拒答/零调用证据并提交。

### Task 4: 图内 ReAct 与真正流式答复

**Files:** Create `app/services/workflow/agent.py`, `app/services/workflow/prompts.py`, `tests/test_workflow_agent.py`; Modify `app/services/workflow/graph.py`.

**Interfaces:** AgentNodes 持有 service依赖，`.decide(state)->dict`, `.execute(state)->dict`, `.answer(state)->dict`，由 graph 固定条件边连接；模型 `.bind(max_tokens=n)`、`.bind_tools(read_only_tools).invoke(messages)` 决策，`.stream(messages)` 最终答复。

- [ ] 写一次物流工具收敛、订单→物流两步、缺订单号澄清、多工具消息配对、伪造写工具零 tickets、未知工具/重复调用/预算/超限、非法完成 JSON、真实流中途错误测试。
```python
def test_stream_is_delivered_before_next_model_chunk(workflow, streaming_model):
    events = workflow.stream_events(ChatRequest(conversation_id='s',message='订单1001物流'))
    first = next(e for e in events if e['event']=='token')
    assert first['data']['content'] == '正在'
    streaming_model.release_next.set()
    assert list(events)[-1]['event'] == 'done'
```
- [ ] 专项先红；Context7核对工具绑定/usage与流；实现决策JSON只给next/actions/missing信息，工具观察回填，最终无工具流真实逐块转发，所有模型调用计入预算。
- [ ] 失败或达到边界时不能遗留未匹配tool calls进入后轮；停止生成固定答复，日志原因明确。证据作为数据，正经返回引用但不追加后验闸。
- [ ] 路由/Agent专项全绿，记录两步 trace 与流式证据并提交。

### Task 5: API 默认切换与生命周期

**Files:** Modify `app/main.py`, `app/schemas.py`, `app/services/quality/runtime.py`; Create `app/services/workflow/runtime.py`, `tests/test_workflow_api.py`; Update `tests/test_chat_api.py` 首页断言。

**Interfaces:** `build_workflow_service(settings)->WorkflowService`；新增 TicketRequest(conversation_id,user_id,message_id)；`POST /api/v1/tickets` 返回 ticket_no/status，不通过聊天文本触发；create_app保留注入chat_service测试接口。

- [ ] 写默认原生首页（即使dist存在）、SSE事件协议、ticket入口归属/未建议拒绝/重复请求、service close lifespan 测试。
```python
def test_default_root_is_native():
    client = TestClient(create_app(chat_service=FakeChatService([])))
    page = client.get('/')
    assert page.status_code == 200
    assert 'id="chat-form"' in page.text
    assert '/src/main.jsx' not in page.text
```
- [ ] 先红，Context7核对 StreamingResponse/lifespan，默认工厂用官方 saver和ch04 retriever（直接复用factory构造结果的retriever，不执行旧answer），QUALITY_ENABLED=false复用dense。
- [ ] lifespan关闭checkpoint与engine，纯闲聊/投诉不因没有embedding服务连接而初始化检索器；保留source/after-sale路由。
- [ ] API与既有chat专项通过，记录并提交。

### Task 6: Luna 原生页面独立按钮与兼容展示

**Files:** Modify `app/static/index.html`; Create `tests/browser/ch05.spec.mjs`（若环境无浏览器测试库，使用独立验证脚本与真实浏览器工具）；仅涉及该页面与前端行为测试。

**Interfaces:** actions事件 `{items:[{type:'handoff'|'create_ticket',label:string}],message_id:int}`；ticket JSON请求 `{conversation_id,user_id:'demo-user',message_id}`，成功返回 `{status:'created',ticket_no}`。citations已有items快照；done包含message_id。

- [ ] Luna先写交互用例并观察原页面不能满足两个按钮而红：取消无fetch；handoff两句固定展示且零ticket请求；ticket确认仅自身按钮禁用，重复不多写；都不点继续send；旧按钮不能串入新回复。
```javascript
await page.getByRole('button', {name:'转人工', exact:true}).click();
// 确认后两条模拟信息出现；独立建单按钮仍可点；ticket请求计数仍为0。
await expect(page.getByText('已转接人工客服', {exact:true})).toBeVisible();
await expect(page.getByText('您好，我是客服小猫，请问有什么可以帮您的', {exact:true})).toBeVisible();
```
- [ ] 实现独立确认交互与状态，SSE actions逐条绑定，转人工只本地；render引用/受限原文链接、本地满意度保留，不扩大工作台功能；使用textContent避免模型HTML注入。
- [ ] 浏览器交互全绿后主智能体验证API接线，记录并提交。

### Task 7: 真实 prompt 评估、五项演示与分支收尾

**Files:** Create `scripts/evaluate_ch05.py`, `scripts/demo_ch05.py`, `evaluation/ch05/cases.json`, `tests/test_ch05_evaluation.py`; Modify `README.md`, `dev-notes/ch05.md`.

**Interfaces:** `python3 -m scripts.evaluate_ch05 --fixture|--live` 输出独立run JSON/Markdown，包含分类JSON失败/路由/工具步数/预算等；`python3 -m scripts.demo_ch05 --fixture|--live` 输出五项验收trace。真实请求/响应保存但不保存key。

- [ ] 先建立标注样例和验收fixtures；评估指标/错误独立分类用TDD验证，prompt 本身按用户要求以标注样例 live 验证替代单测。
```python
def test_invalid_classification_is_error_not_success():
    report = summarize([{'expected':'投诉','actual':None,'error':'invalid_json'}])
    assert report['correct'] == 0
    assert report['errors'] == 1
```
- [ ] 运行裸循环演示、图fixture、真实配置prompt评估和服务烟测；真实数据库/知识服务不可用如实报告，fixture不能代替live。
- [ ] 完整 `python3 -m pytest -q`，前端交互测试，`git diff --check`，验证迁移与资源关闭；独立fresh reviewer对全分支评审，Critical/Important必须修复并红绿验证。
- [ ] 使用 finishing-a-development-branch；按用户授权保留已验证分支，不自动合并/推送。交付命令、测试结果、代码位置和阶段笔记。主仓库 dev-notes/ch05.md逐阶段镜像，避免只在工作树里有记录。
