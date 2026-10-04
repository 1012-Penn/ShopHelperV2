# MewHelp 电商客服知识库

客服系统使用 Milvus 原生 BM25（内置 chinese analyzer）与 BGE-M3 dense 双路各召回 Top-50，hybrid_search + RRF 融合后由 `BAAI/bge-reranker-v2-m3` 精排 Top-10。MySQL `knowledge_chunks` 保留权威原文；第5章在线路径由固定Workflow强制检索并先过最简证据闸，再交给ReAct Agent流式回复；缺证明确拒答并记录原问题。第4章缓冲生成与引用校验仍供原评估服务使用。聊天页角标可查看当轮原文与章节路径，满意度反馈只写浏览器本地。随仓库提供受控电商业务基线，正式门店仍需导入已审核政策。

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

当前随仓库的 4 条小评估集在阈值 `0.60` 下得到正确答案和来源 `3/3`、无关问题误报 `0/1`，因此配置默认值为 `FAQ_MIN_SIMILARITY=0.60`；`0.30` 时无关问题误报 `1/1`。该小样本只验证基线，不足以代表所有门店和真实用户问题。扩充门店审核后的正负样例后重新评估，并将所选阈值写入 `.env`，重启应用生效。

## 验证

```bash
python3 -m pytest -q
docker compose config
```

## 客服工作台与售后识别

项目保留历史 React + Vite 客服工作台源码；第5章 FastAPI 默认首页服务 `app/static/index.html` 原生聊天页，不再优先加载 dist。原 React 工作台可在 Vite 开发环境单独运行，构建命令保留：

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


## ch04 演示与四策略评估

启动前迁移并重建 `HYBRID_COLLECTION=knowledge_ch04`；旧 dense 集合保留。设置 `QUALITY_ENABLED=false` 可回到 ch03 路径。新知识/正文改动后可以继续使用原 build/extract/sync 命令；按集合串行同步，基于独立 content_hash checkpoint 恢复。

```bash
python3 -m scripts.migrate_ch04
python3 -m scripts.rebuild_hybrid_index
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

访问 http://127.0.0.1:8000，问“邮费是多少”后点击 [n] 查看原文及章节，点击 👍/👎 只会在 localStorage 的 `mewhelp.feedback.v1` 留一条记录并锁定。问“请告诉我今晚彩票中奖号码”得到明确拒答；数据库 low_confidence_questions 保存用户原话和数值会话外键。品类筛选可在聊天 POST JSON 中加入 `category`，两路均先过滤。例如：

```bash
curl -N http://127.0.0.1:8000/api/v1/chat/stream   -H 'Content-Type: application/json'   -d '{"conversation_id":"ch04-demo","message":"邮费是多少","category":"配送与运费 / 运费与配送范围"}'
```

category 必须使用数据库实际 `knowledge_chunks.category` 值；未匹配品类会按无证据拒答。

评估集 300 条，五桶各 60，难度三档；60 条校准与 240 条 test 分别报告。评估使用独立 SQLite `evaluation/ch04/eval.db` 和 Milvus `knowledge_ch04_eval`，其中 XH 测试型号全部虚构，只供检索验收，不入线上集合。可通过 EVAL_DATABASE_URL/EVAL_HYBRID_COLLECTION 自定义；faith_cases 持久台账仍写主业务 MySQL。

```bash
python3 -m scripts.validate_ch04_dataset
python3 -m scripts.evaluate_ch04 --fixture
python3 -m scripts.evaluate_ch04 --live --split all --workers 12 --calibrate
python3 -m scripts.faith_cases --status 未解决
python3 -m scripts.faith_cases --eval-id D046 --status 已解决 --resolution '填写实际完成的修复和验证依据'
```

fixture 仅验证指标/接线，不能替代真实质量结果。每轮报告以 run_id 独立保留 JSON/Markdown/逐题证据，不覆盖历史。默认裁判与生成使用同一聊天模型，有自评偏差；可以独立配置 JUDGE_MODEL/JUDGE_API_KEY/JUDGE_API_BASE。拒答忠实度记 N/A，同时单列已知问题误拒答、未知拒答、错误、空答率和指标分母。faith_cases 默认只长期收录 hybrid_rerank 编造个案，四策略所有个案均保留在报告；反复判出会累加计数、重新打开，未再次判出不自动解决。

2026-10-01 最终实跑：300×4=1200 组，无调用错误；[完整报告](evaluation/ch04/runs/20261001-010751-f0400f2a/report.md)。

| 策略 | Recall@10 | MRR | Faithfulness（逐题宏平均） |
|---|---:|---:|---:|
| dense | 98.54% | 0.896 | 1.000 |
| BM25 | 93.96% | 0.797 | 0.998 |
| hybrid | 99.17% | 0.881 | 1.000 |
| hybrid_rerank | 99.79% | 0.944 | 1.000 |

BM25 的 60 道型号题 Recall@1=100%。最终 hybrid_rerank 未知题拒答率 100%，已知题误拒答率 10.42%（独立 test 为 9.38%），检索排名提升并不意味着所有问题都能回答。RERANK_MIN_SCORE=0.05 来自 60 条校准题，需针对真实店铺重新校准。数据由助手构建，未经业务专家独立复核，跨桶相同政策仍可能语义相关。

开发过程与独立评审修复详见 [dev-notes/ch04.md](dev-notes/ch04.md)。

2026-10-02 新增 [v2 挑战集](evaluation/ch04/v2/README.md)：300 题、480 个有效 chunk、140 道品类过滤题。近似型号、渠道/地域/状态例外、口语问法及 2—3 条必要证据分别标注；全局场景家族不跨 calibration/test。v1 题库和历史报告保留，两个版本默认使用独立 SQLite、Milvus 集合和报告目录。

```bash
python3 -m scripts.validate_ch04_dataset --dataset-dir evaluation/ch04/v2
python3 -m scripts.evaluate_ch04 --live --dataset-dir evaluation/ch04/v2 --split all --workers 8
# 填入实际运行目录后，仅重新判分，保留原生成答案和证据
python3 -m scripts.evaluate_ch04 --judge-only evaluation/ch04/v2/runs/<run_id>/rows.jsonl --workers 8
# 已配置 JUDGE_MODEL/JUDGE_API_BASE/JUDGE_API_KEY 后，可强制检查独立裁判身份
python3 -m scripts.evaluate_ch04 --judge-only evaluation/ch04/v2/runs/<run_id>/rows.jsonl --require-independent-judge
```

v2 报告区分 Recall@1/5/10、完整证据@K、MRR、必要事实覆盖/答案正确性、Faithfulness 和拒答质量，并给 test 同题配对差异。正确性裁判可看必要事实标注；忠实度裁判仍只看当轮证据，生成模型不接触 ground-truth。正例误拒答正确性为 0，未知题明确拒答正确性为 1；裁判失败单列错误和有效样本数。冻结后的题库/语料 hash 不一致会拒绝运行，防止按 test 成绩删题或改标注。不同难度的数据集不能直接按总分比较是否退化；当前裁判仍是同模型基线，未做业务专家盲审。

2026-10-02 v2实跑完成，1200组、错误0：[报告](evaluation/ch04/v2/runs/20261002-120853-23796104/report.md)、[结果解读](evaluation/ch04/v2/runs/20261002-120853-23796104/analysis.md)。dense/BM25/hybrid/hybrid_rerank完整证据@10分别91.67%/75.00%/91.67%/97.08%，必要事实与拒答正确率93.33%/78.67%/93.00%/96.33%。多证据桶完整证据命中由66.67%提升到88.33%；整体MRR最高为hybrid，重排不是所有指标最优。test子集和同题胜/平/负见报告，当前仍为同模型裁判基线。

生成结构解析或引用编号校验失败时，仍明确拒答，并保存当轮原始输出、实际送入模型的证据全集、具体失败谓词和会话/策略标识。默认快照在 `.runtime/quality/generation-failures.jsonl`；可用 `QUALITY_GENERATION_DIAGNOSTICS_PATH` 调整路径。该目录不进入 Git，快照不会发给聊天前端。正常回答和证据不足的自评拒答不记作协议失败。

```bash
# 确定性复现协议异常，核对快照与拒答行为
python3 -m pytest -q tests/test_quality_generation_diagnostics.py
```

2026-10-02 修复阶段：Query 改写保护型号、带单位/正负号数量、模糊时长与否定子句；无法确认保真则保留原话。同义词仍仅检索时扩展。独立 calibration 成对验证后，混合检索融合输出上限采用100（dense/BM25每路仍Top-50，最终bge-reranker-v2-m3精排Top-10）。该上限增加重排工作量；校准实验见 `evaluation/ch04/v2/retrieval-calibration/20261002/report.md`。新评估报告metadata记录Prompt及实现hash与实际融合上限，旧报告保留原配置。

2026-10-02 修复后v2统一实跑：1200组、调用错误0，hybrid_rerank完整证据@10=0.988、正确率=0.980（test=0.979），已知误拒答6/240；其中3条引用声明不一致、2条精排丢必要证据、1条候选缺证。四策略还有6条安全协议拒答，raw已记录，不能等同于生成无问题。详见 [修复后报告](evaluation/ch04/v2/runs/20261002-171650-a36a7908/report.md) 与 [独立结果解读](evaluation/ch04/v2/runs/20261002-171650-a36a7908/analysis.md)；历史基线保留。代码版本 `f135940`，评估显示3位，rawJSON保留精度。

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

28条标注样例是本章基线，fixture只能证明评估接线，不能代表真实prompt效果。开发过程按阶段记录于 `dev-notes/ch05.md`，设计与计划位于 `docs/superpowers/specs`、`docs/superpowers/plans`。

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
