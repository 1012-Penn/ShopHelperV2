# MewHelp 电商客服知识库

默认客服采用 LangGraph 固定 Workflow：指代透传、七类意图、四路分流、知识检索及前置置信度闸、主力 ReAct Agent、日志。检索复用 Milvus BM25 与 BGE-M3 双路召回及重排；知识问题证据弱则先兜底，业务查询由 Agent 调用既有工具。原生聊天页提供独立确认的转人工与建工单按钮。业务工具和转人工仍为演示模拟，正式门店需导入已审核政策。

## 配置和启动

需要 Python 3.10+、Docker Compose、MySQL 和可访问 SiliconFlow `BAAI/bge-m3` embeddings 的 API key。复制环境模板并配置 `DATABASE_URL`、`EMBEDDING_API_KEY`、`MILVUS_URI` `RERANK_API_KEY`（SiliconFlow 固定重排模型）以及客服聊天模型的 `MODEL`、`API_KEY`、`BASE_URL`。嵌入密钥只放在本机 `.env` 或受控环境变量中，不要提交到版本控制。

```bash
python3 -m pip install -e '.[dev]'
cp .env.example .env
# 编辑 .env 后启动基础服务和 Milvus
docker compose up -d db etcd minio milvus
```

首次建库会读取 `knowledge_docs/*.md` 和当前 FAQ 表，先写入 MySQL，再将 pending 块向量化写入 Milvus。命令可安全重复执行；中断后重新运行会按 MySQL 主键补齐 pending 向量。

```bash
python3 -m scripts.seed
python3 -m scripts.migrate_ch04
python3 -m scripts.build_knowledge
python3 -m scripts.rebuild_hybrid_index
```

运行客服 Web 应用：

```bash
uvicorn app.main:app --reload
```

打开 <http://127.0.0.1:8000/>，输入“邮费是多少”或“运费怎么查”，应从配送/运费知识块回答，并以实际结算页为准。订单、商品和物流工具仍按项目现有业务实现运行。

## 对话知识抽取与恢复

对话抽取按批读取已落库的客服消息，对消息脱敏后调用配置的聊天模型抽取问答，候选先进入 staging；批次 checkpoint 和 staging 同事务提交，随后全局去重并写入 `knowledge_chunks`。每次运行会先恢复未提升的 staging 项，再处理新对话。抽取任务使用 `MODEL`、`API_KEY`、`BASE_URL`；只运行文档/向量任务时无需配置聊天模型密钥。

```bash
python3 -m scripts.extract_conversation_knowledge --batch-size 100
python3 -m scripts.sync_knowledge_vectors --batch-size 32
```

向量恢复命令只处理一批 legacy pending 块，并在 `QUALITY_ENABLED=true` 时同步启用中的 hybrid 集合；文档建库与对话抽取也维护新集合。以下 cron 为示例。legacy 这批，若部分失败会返回非零退出码。系统 cron 可以按业务频率调度，例如每小时抽取一次、每 5 分钟补向量：

```cron
0 * * * * cd /path/to/MewHelp && .venv/bin/python -m scripts.extract_conversation_knowledge --batch-size 100 >> /var/log/mewhelp-knowledge-extract.log 2>&1
*/5 * * * * cd /path/to/MewHelp && .venv/bin/python -m scripts.sync_knowledge_vectors --batch-size 32 >> /var/log/mewhelp-knowledge-sync.log 2>&1
```

## 标注集评估

将 `tests/fixtures/faq_cases.json` 中的预期命中来源和答案片段作为小型回归评估集。评估脚本对 Top-5 dense 结果测试阈值，输出正确来源/答案命中数和无关问题误报数；只有全部正例正确且负例无误报时才选择候选阈值。当前标注集很小，新增门店前应使用该门店已审核语料扩充正负样例后再设生产阈值。

```bash
python3 -m scripts.evaluate_knowledge
```

## 验证

```bash
python3 -m pytest -q
docker compose config
```

## 客服工作台与售后识别

项目同时提供 React + Vite 客服工作台，FastAPI 会托管 `frontend/dist` 中的构建产物。构建前端：

```bash
cd frontend
npm ci
npm run build
cd ..
```

根页面支持多轮客服对话、SSE 流式回复和快捷提问；对话由后端按 `conversation_id` 持久化。右侧售后识别区调用 `POST /api/v1/after-sale/extract`，从描述中提取订单号、诉求类型和期望方案。

可用以下命令运行售后识别的离线样例评估和服务烟囱测试：

```bash
python3 -m scripts.evaluate_after_sale --fixture
./scripts/smoke_test.sh
```

烟囱测试会启动 FastAPI 并调用配置的聊天模型，因此需要本地 `.env` 中有可用的模型配置。


## 混合检索与回答质量

每路召回 Top-50，RRF 融合后使用固定模型 `BAAI/bge-reranker-v2-m3` 精排 Top-10。证据不足时明确拒答，并写入 `low_confidence_questions`；回答引用可点击查看原文和章节路径。满意度反馈保存在浏览器本地。

知识库是受控演示资料，真实部署前应换成经过审核的商家政策。演示不能保证每个复杂问题都回答完整；多事项证据遗漏、场景匹配与模型引用错误仍需按实际业务验证。

## 四策略评估

提供两个各 300 题的固定评估集，比较 dense、BM25、hybrid、hybrid_rerank。v2 包含近似型号、渠道与地域例外、口语问法、多证据问题及未知问题。评估语料只进入隔离 SQLite 和 Milvus 集合。数据与标注为受控虚构场景，未经业务专家独立复核。

```bash
python3 -m scripts.validate_ch04_dataset --dataset-dir evaluation/ch04/v2
python3 -m scripts.evaluate_ch04 --fixture --dataset-dir evaluation/ch04/v2
python3 -m scripts.evaluate_ch04 --live --dataset-dir evaluation/ch04/v2 --split all --workers 8
```

fixture 验证接线和指标计算，不能代表真实效果。真实评估会调用配置的 embedding、rerank、生成与裁判服务，产生 API 费用。报告在本地 `evaluation/ch04/v2/runs/` 生成，不进入版本控制。

独立裁判可通过 `JUDGE_MODEL`、`JUDGE_API_BASE`、`JUDGE_API_KEY` 配置；留空时复用生成模型，存在同模型评判偏差。评估展示 Recall@K、MRR、完整证据覆盖、答案正确性、Faithfulness 与拒答质量；Faithfulness 高不代表答案完整或业务判断正确。

编造个案记录在业务库 `faith_cases`，可查询并记录处理结果：

```bash
python3 -m scripts.faith_cases --status 未解决
```

## 凭据与本地数据

模型密钥仅填写在本机 `.env` 或环境变量中。`.env.example` 中密钥均为空。Compose 的数据库和对象存储账号为本地演示默认值，部署前自行更改。不要提交真实客户对话、数据库备份、运行日志或模型原始响应。


## ch05：固定 Workflow 与主力 Agent

默认服务采用 LangGraph StateGraph：指代透传→七类意图→四路固定分流。商品咨询/退款退货强制先检索，再用检索得分过闸；物流/订单/售后直接进入 ReAct Agent。投诉固定安抚并给两个独立操作建议；常见问候零模型，其他闲聊最多分类一次。决策、工具执行、最终答复分别是图节点，最终答复真实流式输出。

先演示不依赖编排框架的普通循环，再运行图与页面：

```bash
python3 -m pip install -e '.[dev]'
python3 -m scripts.demo_bare_agent --fixture
python3 -m scripts.demo_bare_agent --live
python3 -m scripts.migrate_ch05
python3 -m scripts.demo_ch05 --fixture --output-dir .runtime/ch05/demo-fixture
python3 -m scripts.build_knowledge
python3 -m scripts.demo_ch05 --live
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

打开 http://127.0.0.1:8000/，依次输入“退货政策是什么”“订单1001的物流到哪了”“我要投诉”“你好”“先查订单1001状态，再查询它的物流轨迹”。图演示将五项路径、工具调用、预算用量与答案写入 `evaluation/ch05/runs`；运行日志默认 `.runtime/ch05/workflow.jsonl`，官方 SqliteSaver 状态默认 `.runtime/ch05/checkpoints.sqlite`。知识初始化延迟到知识分支，问候/投诉不连接 Milvus。

投诉后点“转人工”并确认，仅页面显示“已转接人工客服”和“您好，我是客服小猫，请问有什么可以帮您的”，不接真人系统、不写 tickets。点“建工单”并确认，独立 POST `/api/v1/tickets` 才调用原工单工具。可以只点一个、两个都点或都不点继续聊天。创建成功重复请求返回同一单号；处理中或结果未知不能自动重试，页面提示核查。

默认限制：4次工具决策、6次只读工具执行、每模型调用最多512输出 tokens、当轮模型累计12000预算（检索器内部归一化调用由原服务执行，账单用量不包含在此数字内）。`.env.example` 提供 checkpoint、日志路径与预算参数。日志区分实际usage与保守估算，不显示内部推理。

此部署演示为单进程SQLite checkpoint；真实多副本部署需要另行确定后端与并发方案。订单/商品/物流工具沿用第2章的模拟数据；人工转接也仅模拟。本章没有实现正式指代/意图模型、上下文升级、MCP业务接入或飞轮入库。知识闸是检索分数的最简保护，不等于正式证据覆盖度/生成正确性保证。

验证与真实prompt评估：

```bash
python3 -m pytest -q
python3 -m scripts.evaluate_ch05 --fixture
python3 -m scripts.evaluate_ch05 --live
# 独立浏览器测试依赖，不影响产品使用原生JS
npm install --prefix .runtime/browser playwright
.runtime/browser/node_modules/.bin/playwright install chromium
node --test tests/browser/ch05.spec.mjs
```

28条标注样例是本章基线，fixture只能证明评估接线，不能代表真实prompt效果。

本次真实验收使用独立 Milvus collection，避免覆盖已有第4章索引：

```bash
MILVUS_COLLECTION=knowledge_ch05_demo HYBRID_COLLECTION=knowledge_ch05_demo_hybrid python3 -m scripts.build_knowledge
MILVUS_COLLECTION=knowledge_ch05_demo HYBRID_COLLECTION=knowledge_ch05_demo_hybrid python3 -m scripts.demo_ch05 --live
MILVUS_COLLECTION=knowledge_ch05_demo HYBRID_COLLECTION=knowledge_ch05_demo_hybrid python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

## 第 7 章：当前会话上下文

当前会话采用三层历史：L1按完整轮保留原文，L2保留用户原话、客服前60字及工具标记，最早部分由后台任务追加到 `conversation_summaries`。SQL原文和LangGraph checkpoint不裁剪；新工具观察只留checkpoint。历史与精简模型输入分别保存，不建立用户画像或跨会话记忆。

首次启动会校验预算；已有MySQL表通过幂等增量迁移补两个锚点，新摘要表单独创建：

```bash
python3 -m scripts.migrate_ch07
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开聊天首页，侧栏可新建或切换旧会话，回载原文继续聊。只读接口为 `GET /api/conversations?user_id=demo-user` 和 `GET /api/conversations/{id}/messages?user_id=demo-user`；沿用本项目既有user_id归属约定。侧栏加载失败不会阻断发送。

预算默认 `MODEL_CONTEXT_WINDOW=128000`；窗口余量扣除系统/工具2200、检索 `RERANK_TOP_K×150`、摘要1000、输出2000、安全500，以及 `MAX_USER_INPUT_TOKENS + MAX_AGENT_STEPS×(TOOL_RESULT_MAX_TOKENS+100)` 的瞬时峰值。历史取目标40轮×稳态2000与窗口余量的较小值。L1/L2按70/30分配，L1另留1token边界余量。中文估算与所有预算共用 `CHINESE_TOKEN_RATIO`，默认1.0保守估计；真实usage的校准报告可用下面评估命令生成，调整时应同时评审稳态/峰值和预留参数。旧 `AGENT_MAX_DECISIONS`/`AGENT_MAX_OUTPUT_TOKENS` 仍作为新变量未设置时的兼容别名；`AGENT_MAX_TOKENS` 单独控制累计调用费用，默认1000000。

演示配置和可复现命令：

```bash
# 不调用外部API的25轮机制演示，输出目录必须是新目录
python3 -m scripts.demo_ch07
python3 -m scripts.demo_ch07 --demo
# 真实配置模型跑25轮完整图（订单/物流工具仍是项目的模拟数据）
python3 -m scripts.demo_ch07 --live
python3 -m scripts.demo_ch07 --live --demo
# 真实模型标注评估：摘要、指代、长历史分类、重复信息保真
python3 scripts/evaluate_ch07.py --env-file .env
pytest -q
CH07_LIVE_MYSQL=1 pytest tests/test_ch07_mysql.py -q
```

`--demo` 使用以下预算配置，算得历史5650、L1=3954、L2=1695；实际聊天页面可按同组环境变量启动：

```bash
MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 \
MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 \
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

每轮完整上下文记录在 `log/app.log`；演示独立记录在输出目录的app.log，运行结果json包含路径。可直接检查：

```bash
rg 'model_ctx|history_ctx|层1 降级|summary trigger|summary start|summary done|summary skip|summary fail' log/app.log
```

后台摘要只追加段，不重写旧段；长积压按完整轮分批，失败不推进边界、下一轮可重试。所有摘要段永久保留在数据库，模型只注入摘要预算内的最新完整段；极长会话的最早摘要可能不再注入，日志注明省略段数。真实原文仍能完整回载。输出和工具输入上限与窗口实际校验共同保护预算，原始超大工具结果仍留checkpoint。

本章开发过程、失败实测与独立评审记录见 `dev-notes/ch07.md`；标注结果保存在 `evaluation/ch07/runs/`，fixture机制演示不代表真实模型质量。
