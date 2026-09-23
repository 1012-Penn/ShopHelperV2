# 第二章：客服 Function Calling 设计

## 目标

在第一章的纯对话基础上，建立可运行的电商客服数据查询切片：客服聊天通过 LangChain Function Calling 选择至多一个业务工具，执行后将结果回灌模型并逐 token 输出最终回答；聊天消息与工具往返记录保存在 MySQL。

## 已确认的设计决策

- 用户确认采用独立聊天编排服务与工具注册/执行层，HTTP 路由只负责请求转换和 SSE 事件输出。
- 用户确认由 SQLAlchemy ORM 定义本章数据库结构，不要求预先提供建表 SQL；如需兼容其他现存数据库规范，另行提供 DDL。
- 保留第一章的 FastAPI、LangChain、OpenAI-compatible 模型接入和 `POST /api/v1/chat/stream` 入口。
- Function Calling 只允许单次工具执行，不实现自动 Agent loop。
- 聊天页按 Vibe Coding 方式改造，不对聊天页套用单独的 brainstorm、TDD、code review 门槛；聊天服务和工具业务逻辑仍按项目 TDD 规范实现。

## 范围

包含：

- FastAPI + SQLAlchemy 的分层项目骨架，以及 Docker Compose MySQL。
- `faq`、`conversations`、`messages`、`tickets` 四张表和演示 FAQ 数据。
- 五个 LangChain `@tool` 工具：`query_order`、`query_product`、`query_logistics`、`query_faq`、`create_ticket`。
- 工具注册管理、输入 Schema 校验、执行错误转换、超时与有限重试。
- 现有 SSE 聊天入口内的工具选择、一次工具执行、结果回灌和最终流式回答。
- 聊天记录中保存用户消息、assistant 工具申请、tool 结果及最终 assistant 回答。
- 聊天气泡中的工具徽章与工具执行状态。
- 自动化测试、FAQ 标注样例和本地演示说明。

不包含：

- 订单、商品或物流真实 API 对接，也不为它们建数据库表；三个工具仅生成演示结果。
- 多次/多步自动工具调用、Agent loop、向量检索、RAG。
- 登录认证、权限系统、生产级数据迁移框架。

## 架构

### HTTP 层

`app/main.py` 保留 `POST /api/v1/chat/stream`，验证请求并将服务事件编码为 SSE。已有的聊天页面使用该入口，显示 `tool_status` 状态、工具徽章、`token` 文本和 `done` 完成态。

### 聊天编排层

`app/services/chat.py` 接收会话 ID、用户 ID 和消息文本，协调数据库持久化、模型工具选择、工具执行、结果回灌和最终回答流。模型配置继续复用第一章的 OpenAI-compatible 配置，不在应用代码中硬编码密钥。

每轮流程：

1. 查找或创建会话，写入 user 消息并提交短事务。由于本章不做认证，聊天请求提供可选 `user_id`，缺省使用演示用户 ID。
2. 以客服 System Prompt 和本轮用户消息调用已绑定五个工具的模型流。服务端收集本次模型响应的内容片段和工具调用增量，确定响应种类后再发送对应 SSE 事件。
3. 若无工具调用，按模型返回的文本片段发送 `token` 事件并保存 assistant 消息。
4. 若有且仅有一个工具调用，保存包含调用名称、参数和 `tool_call_id` 的 assistant 消息，发送 `tool_status` SSE 事件，执行工具并保存 `role=tool` 结果。
5. 将含工具调用的 assistant 消息和匹配 `tool_call_id` 的 `ToolMessage` 一起回灌模型，使用流式接口生成最终回答；每个文本片段作为 `token` 输出，聚合完成后保存 assistant 消息并发送 `done`。

模型返回多个工具调用时不执行其中任何一个，将其视为模型响应不符合单次调用约束，并走明确错误处理。不会在任何情况下启动循环继续请求工具。

### 数据层

SQLAlchemy 声明式 ORM 模型负责建表，数据库会话使用短事务；不在等待模型或外部工具时持有数据库事务。Docker Compose 启动 MySQL；初始化/seed 命令可重复执行且不会重复插入相同 FAQ 样例。

字段至少包括：

| 表 | 字段 |
|---|---|
| `faq` | 主键、问题、答案、分类、创建时间 |
| `conversations` | 会话 ID、用户 ID、处理状态、创建时间 |
| `messages` | 主键、会话 ID、role（`user` / `assistant` / `tool`）、内容、工具调用申请（可空 JSON）、`tool_call_id`（可空）、创建时间 |
| `tickets` | 工单号主键、关联会话 ID、问题描述、工单类型、处理状态、创建时间 |

会话 ID 使用客户端提供的 `conversation_id`；对新 ID 创建会话，对已有 ID 追加消息。工单号由应用生成。FAQ 的关键词查询针对问题字段做参数化 `LIKE` 匹配，不做向量或同义词扩展。

## 工具

| 工具 | 输入 | 行为 |
|---|---|---|
| `query_order` | 订单号 | 返回随机生成的演示订单信息 |
| `query_product` | 商品关键词或编号 | 返回随机生成的演示商品信息 |
| `query_logistics` | 订单号 | 返回随机生成的演示物流信息 |
| `query_faq` | 用户问题/关键词 | SQL `LIKE` 搜索 FAQ；无匹配则返回明确未命中 |
| `create_ticket` | 问题描述、工单类型 | 使用服务端绑定的当前会话 ID 写入 `tickets` 并返回工单号 |

所有工具以 LangChain `@tool` 定义，使用类型标注生成参数 Schema。注册表仅暴露上述五个工具；执行前检查工具名并校验参数。未知工具和参数校验错误不重试。工具的临时执行失败可有限次数重试；超时和不可重试异常转换成结构化工具错误结果，记录后回灌模型，让模型向用户说明处理结果。模型不得提供或覆盖工单的会话 ID。

工具调用上下文通过每轮聊天创建的工具实例/闭包绑定会话 ID 和数据库访问能力；这些内部上下文不作为模型可填参数暴露。

## 持久化与 SSE 契约

消息按对话顺序写入 `messages`：user 消息；若有工具，assistant tool-call 申请；tool 执行结果；最终 assistant 回答。工具申请和结果保存相同的 `tool_call_id`。`tool_calls` 字段以 JSON 形式保存工具名和参数；tool 消息的 role 为 `tool`。

SSE 事件：

- `tool_status`：本轮工具名称和正在执行的状态，供页面显示徽章/状态。
- `token`：最终 assistant 回答文本片段，字段 `content`。
- `done`：本轮完成，包含 `conversation_id`。
- `error`：流开始后的处理失败，包含可展示的通用错误信息。

无工具调用的普通回答也通过模型流的文本片段发送 `token` 事件。为了不在工具选择完成前错误展示尚未确认的回答内容，首次带工具模型流的文本片段在工具/文本响应确定前仅在本轮内缓冲；若确定为普通回答，再按片段发送。工具执行失败属于工具结果，由模型根据该结果组织最终回答；模型调用、数据库或 SSE 流发生不可恢复错误时输出 `error`（若响应流已开始），并保留已成功提交的消息记录。密钥、堆栈和内部异常详情不得发给浏览器。

## 聊天页

保留现有聊天交互和 SSE 消费入口。新增助手轮次的工具轨迹徽章：收到 `tool_status` 后展示所选工具名称和执行中状态；回答到达后保留该轮工具名称作为轨迹。FAQ 未命中由回答正文呈现，界面不声称查询成功。聊天页不承担服务端的会话持久化逻辑。

## 验证

- 数据层测试验证四种 ORM 实体字段、会话消息关系及工单写入。
- 工具测试验证参数 Schema、注册表拒绝未知工具、FAQ 命中/未命中、工单关联当前会话、随机演示数据工具，以及超时/重试和错误结果转换。
- 聊天服务测试使用确定性 fake model 和测试数据库验证无工具流、一次工具调用的状态事件和完整消息流水、`tool_call_id` 对应、工具错误回灌、以及多个工具调用被拒绝。
- API 测试验证 SSE 事件类型、顺序、最终回答 token 和错误帧。
- 标注 FAQ 样例包含“退货政策是什么”应命中、“邮费是多少”因字面关键词不匹配应未命中；该漏召回是预期结果，记录为下一章检索升级样例。
- 人工浏览器验收覆盖“订单 1001 的物流到哪了”和“退货政策是什么”，确认工具徽章与基于返回结果的回答。
- 演示命令包含 Docker 启动、seed、应用启动和测试命令；如果模型凭据可用，再进行真实模型的手工 SSE 验收，但不得记录或打印密钥。

## 最新 API 依据

实现依照 Context7 查询到的官方文档：FastAPI `EventSourceResponse`/`ServerSentEvent` 支持 POST SSE；SQLAlchemy 2.0 `sessionmaker.begin()` 可在上下文结束时自动提交、失败时回滚；LangChain `@tool` 从类型标注生成输入 Schema，`bind_tools` 生成工具调用请求，并通过匹配 `tool_call_id` 的 `ToolMessage` 回灌结果。不得使用本章排除的 Agent 自动循环抽象。
