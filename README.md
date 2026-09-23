# MewHelp Function Calling 客服演示

客服聊天页通过单轮 Function Calling 查询演示业务数据。订单、商品和物流工具返回随机模拟数据；FAQ 使用 MySQL `LIKE` 查询；人工工单写入 `tickets` 表。模型最多执行一次工具调用，不包含 Agent 循环或 RAG。

## 启动

需要 Python 3.10+、Docker Compose，以及兼容 OpenAI Chat Completions 工具调用的模型服务凭据。

```bash
python3 -m pip install -e '.[dev]'
cp .env.example .env
```

编辑 `.env`，设置 `API_KEY`、模型名、`BASE_URL` 和 `DATABASE_URL`。然后启动 MySQL 并初始化 FAQ：

```bash
docker compose up -d db
python3 -m scripts.seed
uvicorn app.main:app --reload
```

打开 <http://127.0.0.1:8000/>，在聊天框输入问题。可以试：

- `订单 1001 的物流到哪了`：展示物流工具徽章和本轮模拟物流结果。
- `退货政策是什么`：通过 FAQ `LIKE` 命中并回答。
- `邮费是多少`：当前字面关键词检索预期漏召回，作为后续 FAQ 检索升级样例。

浏览器会保存并复用会话 ID；聊天记录、工具调用申请和工具结果写入 MySQL。运行 `python3 -m scripts.seed` 可重复初始化 FAQ。

## 验证

```bash
python3 -m pytest -q
docker compose config
```

`tests/fixtures/faq_cases.json` 是 FAQ 标注评估集，预期命中“退货政策是什么”，预期漏召回“邮费是多少”。
