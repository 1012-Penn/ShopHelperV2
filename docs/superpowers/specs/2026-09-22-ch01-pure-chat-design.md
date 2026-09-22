# 第一章：纯对话设计

## 目标

构建电商智能客服系统的第一个可运行切片：一个无状态 HTTP API，支持多轮对话、逐 token 的 SSE 流式输出、按 token 预算裁剪历史消息，以及将售后描述提取为固定 JSON 结构。

## 范围

包含：

- Python + FastAPI HTTP API。
- 使用 LangChain 组织 Prompt 并调用模型。
- 通过环境变量配置 OpenAI-compatible 上游模型。
- 以 DeepSeek 作为本地默认验证目标，使用现有的 `/root/.env` 密钥，但不复制或打印密钥。
- 客户端提供多轮上下文，服务端按可配置的 token 预算裁剪历史。
- 为流式对话 token 和结束状态提供 SSE 事件。
- 使用 `with_structured_output` 和 Pydantic schema 提取售后信息。
- 自动化测试，以及一组用于验证提取 Prompt 的评估样例。

不包含：

- 工具调用、function calling、Agent 和 Agent 循环。
- 基于数据库的会话持久化。
- 认证、限流、聊天前端、检索和业务系统集成。

## 模型接入边界

应用使用 `langchain_openai.ChatOpenAI` 调用 OpenAI-compatible 端点。`MODEL`、`API_KEY` 和 `BASE_URL` 均由环境变量配置。GPT、DeepSeek、Ollama，以及提供 OpenAI-compatible API 的 Claude 端点都可以接入；本章不实现独立的 Claude 原生 API 适配器。

应用代码中不得出现密钥字面量。配置优先读取项目 `.env`；在当前工作区中，如果项目 `.env` 没有对应配置，可以回退读取 `/root/.env` 中已有的 `DEEPSEEK_API_KEY`。`.env` 文件必须加入 Git 忽略规则。

## 架构

`app/main.py` 负责 FastAPI 应用和 HTTP 层转换；`app/config.py` 负责配置；`app/prompts.py` 负责客服角色 Prompt 和提取 Prompt 模板；`app/schemas.py` 负责请求、响应、消息和结构化提取模型；`app/services/chat.py` 负责模型构建、历史裁剪和 token 流式输出；`app/services/after_sale.py` 负责结构化提取。服务端不写入会话状态，只回传请求中的 `conversation_id`，供客户端关联会话。

## Prompt 约束

System Prompt 必须明确助手是电商客服代表。助手应礼貌、简洁、真实，不得编造订单或政策事实；信息不足时要主动询问；要保护个人信息；遇到不确定或敏感情况要升级处理。助手不得声称拥有未接入的工具，也不得假装已经完成订单操作。

## API 契约

### `POST /api/v1/chat/stream`

Request JSON:

```json
{
  "conversation_id": "demo-1",
  "message": "我的订单还没收到",
  "history": [
    {"role": "user", "content": "我想查物流"},
    {"role": "assistant", "content": "请提供订单号。"}
  ]
}
```

响应使用 `text/event-stream`。每个模型文本片段都作为 SSE `token` 事件发送，事件 JSON 数据包含 `content`。流以包含 `conversation_id` 的 `done` 事件结束。模型或校验失败如果发生在开始流式输出之前，返回普通 HTTP 错误；如果发生在流式输出开始之后，则返回 SSE `error` 事件。

### `POST /api/v1/after-sale/extract`

Request JSON:

```json
{"text": "订单 123456 的耳机坏了，我想换货，最好尽快补发。"}
```

Response JSON:

```json
{
  "order_id": "123456",
  "request_type": "换货",
  "expected_solution": "尽快补发"
}
```

缺失字段用 `null` 表示；`request_type` 限制为定义好的枚举值以及 `其他`。

## 上下文策略

客户端发送历史消息。服务端在消息前加入固定 System Prompt，并使用 LangChain 的 `trim_messages`，配置 `strategy="last"`、近似 token 计数器和 `MAX_HISTORY_TOKENS`。裁剪后的历史从 human 消息开始，以 human 或 assistant 消息结束。当前用户消息在裁剪后始终追加并保留。

## 配置

运行时必需配置：

- `MODEL`：上游模型名。
- `API_KEY`：API 密钥；使用 DeepSeek 时可以由 `DEEPSEEK_API_KEY` 提供。
- `BASE_URL`：OpenAI-compatible 基础地址。
- `MAX_HISTORY_TOKENS`：历史消息 token 预算，并提供明确的默认值。

当模型名、密钥或基础地址缺失时，应用应在启动或请求处理阶段明确失败，不得静默切换到不相关的模型提供方。

## 验证

- 单元测试证明历史裁剪会保留 System Prompt 和当前用户消息。
- Service 测试使用确定性的 fake chat model 验证流式输出。
- API 测试验证 SSE 事件格式以及多轮历史消息的传递。
- 结构化输出测试使用确定性的 fake structured model 验证 Pydantic 解析。
- 在凭据可用时，用一组带标注的售后样例跑通提取 Prompt/model 链路；记录结果时不得记录凭据或完整敏感输入。
- 手工验收命令覆盖 curl 流式回复、携带第一轮历史进行第二轮提问，以及结构化 JSON 提取。
