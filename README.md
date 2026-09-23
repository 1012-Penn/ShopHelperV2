# MewHelp 第一章：纯对话

这是电商智能客服系统的第一章，先跑通无状态纯对话：多轮历史、SSE 逐 token 输出，以及售后描述结构化提取。

## 安装

项目要求 Python 3.10+。在项目目录执行：

```bash
python3 -m pip install -e '.[dev]'
```

复制 `.env.example` 为项目 `.env`，填写模型名和 OpenAI-compatible 地址：

```dotenv
MODEL=deepseek-chat
BASE_URL=https://api.deepseek.com/v1
MAX_HISTORY_TOKENS=1024
```

`API_KEY` 优先从项目 `.env` 读取；本工作区在项目配置缺失时也会回退读取 `/root/.env` 中的 `DEEPSEEK_API_KEY`。不要把密钥写入代码、日志或 Git。

## 启动

```bash
uvicorn app.main:app --reload
```

启动后打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)，即可使用淘宝风格的客服工作台页面。页面由 FastAPI 直接托管，不需要单独启动前端开发服务器。

## 验收命令

### 1. 查看 SSE 流式回复

```bash
curl -N http://127.0.0.1:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"demo-1","message":"我想查物流","history":[]}'
```

输出应包含多个 `event: token` 和一个 `event: done`。

### 2. 连续两轮对话

第二轮把第一轮的用户消息和客服回复放入 `history`：

```bash
curl -N http://127.0.0.1:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"demo-1","message":"那订单号要在哪里找？","history":[{"role":"user","content":"我想查物流"},{"role":"assistant","content":"请提供订单号，我可以帮你确认下一步。"}]}'
```

### 3. 提取售后结构化 JSON

```bash
curl http://127.0.0.1:8000/api/v1/after-sale/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"订单123456的耳机坏了，我想换货，最好尽快补发。"}'
```

预期包含：

```json
{"order_id":"123456","request_type":"换货","expected_solution":"尽快补发"}
```

## 自动验证

```bash
pytest -q
python3 scripts/evaluate_after_sale.py --fixture
ruff check app tests scripts
./scripts/smoke_test.sh
```

`smoke_test.sh` 会启动本地服务并调用真实上游模型，因此需要可用的 OpenAI-compatible 配置；脚本不会回显环境变量中的密钥。
