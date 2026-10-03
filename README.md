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
