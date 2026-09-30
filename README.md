# MewHelp 电商客服知识库

客服系统使用 Milvus 原生 BM25（内置 chinese analyzer）与 BGE-M3 dense 双路各召回 Top-50，hybrid_search + RRF 融合后由 `BAAI/bge-reranker-v2-m3` 精排 Top-10。MySQL `knowledge_chunks` 保留权威原文；回答通过证据充分性自评与引用校验，缺证明确拒答并记录低置信度问题。聊天页角标可查看当轮原文与章节路径，满意度反馈只写浏览器本地。随仓库提供受控电商业务基线，正式门店仍需导入已审核政策。

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
