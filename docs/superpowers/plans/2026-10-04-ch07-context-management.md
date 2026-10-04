# Ch07 Context Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 当前会话原文保存、三层上下文、异步追加摘要、可观测模型输入与多会话侧栏。

**Architecture:** 独立预算与上下文构造器，SQL摘要仓库和后台worker；现有LangGraph保留完整历史。正式原生首页调用只读历史接口。

**Tech Stack:** LangGraph add_messages/SqliteSaver、LangChain trim_messages、SQLAlchemy/MySQL、FastAPI、原生JavaScript。

**Spec:** docs/superpowers/specs/2026-10-04-ch07-context-management-design.md

## Global Constraints

- 不跨会话，不画像，不历史语义检索，不改既有业务工具。
- messages原文不裁剪；摘要表只追加；新工具结果仅checkpoint。
- 固定system前缀；日志原样记录实际模型输入；当前回复不等待摘要。
- 用户“执行”批准提案后采用主智能体推进、Luna独立前端任务；不再每任务询问。
- 使用原生隔离worktree，不纳入主checkout其他章节的未提交修改。
- 校准基线：系统与工具预留2200，证据每条150×top_k，摘要1000，安全500；峰值=user_input+steps×(tool_result+100决策开销)。默认window128000、目标40轮、稳态每轮2000，历史=min(80000,实际余量)。演示得到5650；70/30分配后L1另留1token边界余量，得到3954/1695。预留可配置并检查实际输入。
- 摘要永久保存全部段，注入只选择摘要预算内完整段，明确记录未注入段数量。

## Review Focus

- 工具协议组不可被trim拆开；完整State不得改成精简版。
- worker生成期间L1推进，不可把未压内容标为已摘要。
- 超长用户原话/工具结果导致预算溢出时可解释地停止或限制模型输入。
- sidebar切换/流式并发以及旧动作绑定会话不得串线。
- 旧数据库迁移幂等，越权历史请求无数据泄漏。

### Task 1: Budget and isolated context construction

**Files:** app/services/context/budget.py, manager.py, logging.py; tests/test_context_budget.py, test_context_manager.py.
**Interfaces:** ContextBudget.from_env(values), history_tokens/layer1_tokens/layer2_tokens; ContextManager.prepare(conversation_id, history), model_messages(state,prompt), history_messages(state,prompt).

- [x] 写测试断言演示5650/3954/1695，默认20轮不裁，中文估算一致，窗口装不下一轮报警。
- [x] `pytest tests/test_context_budget.py tests/test_context_manager.py -q`观察RED。
- [x] 实现公式、trim_messages副本、完整轮分组、L2规则及消息顺序；每条背景用HumanMessage。
- [x] 同命令GREEN，检查history不变和system稳定，阶段追记ch07。

```python
def test_demo_budget():
    b = ContextBudget.from_env(dict(MODEL_CONTEXT_WINDOW='18000', MAX_OUTPUT_TOKENS='2000', MAX_USER_INPUT_TOKENS='2000', MAX_AGENT_STEPS='3', TOOL_RESULT_MAX_TOKENS='1200', RERANK_TOP_K='5'))
    assert (b.history_tokens, b.layer1_tokens, b.layer2_tokens) == (5650, 3954, 1695)
```

### Task 2: Persistence, migration and asynchronous summaries

**Files:** app/db/models.py, session.py; scripts/migrate_ch07.py; app/services/context/repository.py, summary.py; tests/test_context_summary.py.
**Interfaces:** repository.read/advance_layer1/append_summary; SummaryWorker.submit(conversation_id, snapshot), close(); snapshot固定覆盖终点，不依赖连续id。

- [x] 写文件SQLite测试：阻塞summarizer仍立即返回submit，失败不推进，重复范围不追加，推进L1不扩大任务覆盖；迁移重复执行。
- [x] `pytest tests/test_context_summary.py -q`观察RED。
- [x] 添加独立摘要表和两个nullable锚点；worker独立session，生成不持会话锁，事务条件核对后追加；日志全生命周期。
- [x] GREEN并追记；prompt另用标注样例评估，不用测试替代真实模型质量。

```python
def test_failure_keeps_boundary(repository, worker):
    before = repository.read('c1').summary_upto_msg_id
    worker.submit('c1', snapshot)
    worker.close()
    assert repository.read('c1').summary_upto_msg_id == before
```

### Task 3: Workflow integration and read-only conversation APIs

**Files:** workflow/graph.py, agent.py, runtime.py, storage.py, state.py; app/main.py, schemas.py; tests/test_ch07_workflow.py, test_conversations_api.py.
**Interfaces:** store.prepare提供SQL稳定id；State携带独立context_history/summary/current_message_id；service.context可选构造参数，默认创建；两只读GET返回items列表。

- [x] 测试多轮原State持续增长、闲聊每轮history_ctx、Agent每次model_ctx、工具不落库、后台摘要不阻塞；API列表归属和原文。
- [x] `pytest tests/test_ch07_workflow.py tests/test_conversations_api.py -q` RED。
- [x] refer消费共用历史，分类和Agent固定system，模型前窗口检查；输出预留与ReAct上限联动；tool输入截到配置预算但checkpoint保存完整观察。原有累计费用上限与窗口分离。
- [x] GREEN + 既有workflow回归，追记ch07。

```python
def test_history_is_logged_for_greeting(service, request):
    list(service.stream_events(request))
    assert 'history_ctx' in service.context.log.path.read_text()
```

### Task 4: Native chat sidebar (Luna, independent)

**Files:** app/static/index.html; tests/browser/ch07.spec.mjs.
**Interfaces:** GET /api/conversations?user_id=demo-user -> {items:[{conversation_id,preview,has_summary,created_at}]}; GET /api/conversations/{id}/messages?user_id=demo-user -> {items:[{id,role,content,citations,actions,created_at}]}.

- [x] 用browser stub接口写新建保留旧会话、点击回载完整原文继续同id、sidebar失败静默、切换竞态的RED测试。
- [x] 实现侧栏新在前、首问、摘要标记；送消息时禁止切换，过期回载不覆盖新选择；恢复已有引用与动作。
- [x] `node --test tests/browser/ch07.spec.mjs tests/browser/ch05.spec.mjs` GREEN；主智能体复核后追记ch07。

### Task 5: Demonstration, prompt evaluation, review and finish

**Files:** scripts/demo_ch07.py, scripts/evaluate_ch07.py; evaluation/ch07/cases.json; README.md; dev-notes/ch07.md.

- [x] 演示至少24轮：默认无降级；18k级联；最早订单追问；完整原文与checkpoint增长；慢摘要本轮先结束。输出可grep日志、预算和计数。
- [x] 标注评估摘要订单/手机号/诉求保真、指代、矛盾纠正、注入攻击和禁止臆造；真实调用结果留evaluation/ch07/runs，不泄露密钥。
- [x] `pytest -q`及浏览器回归、demo两配置、评估和diff检查；只有真实通过才声称完成。
- [x] 用requesting-code-review派独立审阅，修复重大问题；运行verification-before-completion后记录结论；finish保留分支待用户集成，不推送。

## Review result / execution ledger

主智能体已对spec逐节检查覆盖、接口和竞态；预算取整与摘要注入选择已显式记录。用户“执行”作为继续实施授权。过程以dev-notes/ch07.md逐阶段记录；每任务打勾后推进，最终独立code review无剩余Critical/Important；验证通过后提交至codex/ch07-context，保留worktree，不合并、不推送。
