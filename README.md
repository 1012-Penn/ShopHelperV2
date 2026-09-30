# MewHelp 电商客服知识库

客服系统使用 BGE-M3 向量和 Milvus dense 检索商品 FAQ、配送/运费说明、退换货政策及售后手册。MySQL `knowledge_chunks` 保存权威原文和向量化状态；`query_faq(query: str)` 的工具参数及返回 JSON 保持原契约。当前检索只使用 dense 向量，不含关键词、混合召回或重排。随仓库提供的是通用电商客服基线语料；实际店铺价格、运费、时效、售后承诺和例外规则须导入正式政策。

## 配置和启动

需要 Python 3.10+、Docker Compose、MySQL 和可访问 SiliconFlow `BAAI/bge-m3` embeddings 的 API key。复制环境模板并配置 `DATABASE_URL`、`EMBEDDING_API_KEY`、`MILVUS_URI` 以及客服聊天模型的 `MODEL`、`API_KEY`、`BASE_URL`。嵌入密钥只放在本机 `.env` 或受控环境变量中，不要提交到版本控制。

```bash
python3 -m pip install -e '.[dev]'
cp .env.example .env
# 编辑 .env 后启动基础服务和 Milvus
docker compose up -d db etcd minio milvus
```

首次建库会读取 `knowledge_docs/*.md` 和当前 FAQ 表，先写入 MySQL，再将 pending 块向量化写入 Milvus。命令可安全重复执行；中断后重新运行会按 MySQL 主键补齐 pending 向量。

```bash
python3 -m scripts.seed
python3 -m scripts.build_knowledge
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

第三条命令只处理一批 pending 块，若部分失败会返回非零退出码。系统 cron 可以按业务频率调度，例如每小时抽取一次、每 5 分钟补向量：

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
