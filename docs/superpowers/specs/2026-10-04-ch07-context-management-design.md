# 第 7 章：会话上下文管理设计提案

状态：用户2026-10-04回复“执行”后进入实施；已落实施计划并完成独立代码复审。以下补齐实施时采用的预算与摘要注入口径。

## 目标与边界

只管理同一 conversation_id 内的多轮记忆。保留原文、控制模型输入、异步追加事实摘要；不跨会话记忆、不建用户画像、不做历史语义检索或主题重要度选择。不改现有知识检索业务。原生聊天首页增加会话切换。

验收同时覆盖：默认配置二十轮以上不降级不摘要；18k 演示配置完整级联、摘要后追问最初订单；摘要不阻塞当前回复；模型实际输入可直接 grep；切换会话原文回载且可继续聊。

## 现有基础

- `WorkflowState.messages` 已有 `Annotated[..., add_messages]`，现有 SqliteSaver 按 thread_id 保存完整 State，沿用，不替换 checkpoint 技术。
- `AgentNodes.messages` 当前将动态证据放进 system 并传完整历史，需要分离模型输入构造。
- 分类目前只看本轮，refer 透传；工具调用与结果仍写 messages，需要调整。
- `/` 服务 `app/static/index.html`；历史 React 工作台不是此次默认入口。
- 现有 `Limits.max_tokens` 是每轮累计调用额度，不能当作单次模型窗口；旧 UTF-8 字节估算也不适合直接用于中文窗口预算。

## 实现方式比较

推荐独立 ContextManager + SummaryWorker，在现有图的节点边界准备精简输入，后台服务完成追加摘要；数据库提供原文和边界，checkpoint 提供完整模型消息和工具观察。这样可以单独验证预算、并发和输入顺序。

可选方案是在图内添加专门摘要节点：容易跟踪，但普通顺序节点会阻塞本轮，仍需额外异步调度，不适合作为本章首选。另可引入外部任务队列：有利于多实例恢复，但增加现有栈以外的部署依赖，本章不引入。

## 数据与两个锚点

`conversations` 增加 nullable `summary_upto_msg_id`、`layer1_from_msg_id`。新增 `conversation_summaries`：会话、段序号、覆盖起止 msg_id、事实摘要、创建时间；同一会话同一段覆盖范围唯一。

边界语义：摘要覆盖 `id <= summary_upto_msg_id`，L2 为 `summary_upto_msg_id < id < layer1_from_msg_id`，L1 为 `id >= layer1_from_msg_id`。未摘要时摘要锚点为空；L1 起点按完整轮边界选择。降级只推进 L1 起点，不移动、覆盖或删除 messages 原文。

新消息表只落用户原文与客服最终原文（包含现有引用和操作元数据），新工具观察只留完整 State/checkpoint。存量 tool 行不破坏性删除，侧栏历史不展示内部工具行。稳定消息 ID 映射 SQL ID；工具协议组关联所在轮，使锚点无需单独落工具行仍能确定覆盖范围。

L1 保留包括工具协议组在内的原文，禁止部分消息截断；使用 LangChain `trim_messages` 的独立输入副本、统一 token_counter、`strategy='last'`、`allow_partial=False`，再校验完整轮和工具配对。被移出的完整轮进入 L2。L2 用户原话不变、最终客服答复只保留开头几十字；历史工具调用及结果折成一行普通背景标识，不形成孤立 ToolMessage。不向 State 写 RemoveMessage 或裁剪结果。

## 异步追加摘要

按 L2 实际估算 token 触发，不按消息条数。任务快照固定 `(summary_upto_msg_id, layer1_from_msg_id)` 与完整覆盖输入；摘要覆盖终点为该快照 L1 起点前的最后一条历史消息，不使用 `id-1` 假定会话内连续。

每会话最多一个运行任务；生成调用使用独立模型和独立数据库 Session，不持会话回复锁。成功时短事务核对旧摘要边界，追加一段并推进摘要锚点；如 L1 在生成期间继续推进，新的 L2 留待下个任务，不能误报已摘要。失败不动边界，记录原因、允许后续轮重试；重复覆盖提交跳过。应用关闭时处理 worker 生命周期，避免先关引擎再写摘要。

旧摘要仅作为模型背景，不作为此次待压缩内容，不更新旧段。摘要事实范围仅为已出现的商品、订单号、手机号、明确诉求和未解决问题，禁止补全猜测、禁止保留寒暄；目标几十至一两百字。用标注样例验证标识保真、无臆造、纠正信息、未解决事项和旧摘要不重压。

有限窗口无法无限注入所有追加摘要。实施口径为表内永久保留原段，模型摘要区只注入独立预算可容纳的最新完整段，未注入段数量进入日志；不改写、不合并已存摘要。用户可通过原文回载查看完整会话；极长会话可能不再注入最早摘要，此限制明确记录。

## 模型上下文与 State

主 Agent 顺序为固定 system（人设、红线，配合固定工具定义）、L2 截短历史、L1 原文历史、当前用户原文、单条背景消息（摘要全文与检索证据）。本轮 ReAct 新产生的 AI/tool 协议消息依次追加在背景消息之后；不能移到旧历史或破坏配对。动态 mode/missing 等移出 system。固定决策和回答角色各自维持稳定前缀。

摘要/证据背景使用普通消息，明确标注为非指令数据，不作为 system。完整历史仍经 add_messages 逐节点追加并 checkpoint；精简上下文作为独立字段/值，不能替换 messages。

refer 与 classify 共用同一份摘要+滑窗输入。指代消解只补足明确历史实体，不创造订单或诉求；分类消费消解结果。问候零调用路径仍每轮记录 history_ctx，保持第5章行为。

## 窗口预算与校准口径

统一 estimator 考虑中文字符、其他文本、消息框架及工具 schema；预算和实际日志用同一口径。用标注中文样本/实际 usage 校准系数，禁止只改估算器而不改预算。

定义：`available = window - system_tools - evidence_reserve - summary_reserve - output_reserve - safety_reserve - react_peak`；`history = min(target_turns * steady_turn_tokens, available)`。ReAct 峰值按 MAX_USER_INPUT_TOKENS、MAX_OUTPUT_TOKENS、MAX_AGENT_STEPS、TOOL_RESULT_MAX_TOKENS 推导，并明确工具调用次数/并发工具结果上限；检索预留与 RERANK_TOP_K 联动。每次调用再检查实际输入与输出预留是否越窗，累计调用费用另行统计。

层预算原则为历史的七成/三成，启动打印完整分解，连一轮也装不下报警。超长用户输入或工具结果必须有明确配置上限，不能以 estimator 掩盖真实长度。默认窗口与目标轮数保证代表性二十轮样例不降级，单条极端输入另测溢出保护。

实现采用配置基线：window默认128000，系统+工具2200，证据每项150×top_k，摘要1000，输出2000，安全500；ReAct峰值=user_input + steps×(tool_result+100决策开销)。历史上限为目标40轮×每轮2000。指定18k配置算得18000−2200−750−1000−2000−500−[2000+3×(1200+100)]=5650。L1=history×7//10−1，L2=history×3//10，多留1token边界余量，得到3954/1695。所有开销可配置；保守中文系数1与预算基线一起校准，真实usage评估留下建议而不单独改变默认口径。实际调用另计完整工具定义并校验窗口，累计API费用额度单独配置。

## 日志和会话侧栏

独立 app.log 记录器输出 `log/app.log`。模型调用前写 model_ctx：实际序列化消息、摘要全文、L2/L1逐条原文或规则截短文本、窗口条数、估算token、阶段和会话ID；与真正传给模型的数据一致。history_ctx 每轮在快速路由前写摘要+滑窗，指代和分类调用使用同一快照。

worker 写 summary trigger/start/done/skip/fail：覆盖起止、两个锚点、L2 token/预算、段号、耗时和失败类型；降级写“层1 降级 X→Y”。当前回复不 await 摘要模型，使用可控阻塞的 fake summarizer 验证回复先完成。

`GET /api/conversations?user_id=...`：该用户会话，created_at/id 稳定倒序，首问预览、已摘要标记。`GET /api/conversations/{id}/messages?user_id=...`：归属校验后按id回载用户/最终客服原文及引用/操作元数据；不存在404、越权403。不扩展认证体系，沿用项目现有user_id约定。

侧栏加载失败静默；新对话生成新ID，旧会话保留。切换回载处理过期请求与正在流式回复的竞态，继续沿用选定 conversation_id；消息展示使用 textContent，恢复引用、动作及message_id。

## 验证与交付

代码部分按TDD：预算公式/边界/完整历史不变/工具配对/worker快照和失败幂等/API归属和原文。Prompt部分使用标注集实测指代与摘要事实保真，fake模型只验证机制，不能替代真实生成评估。

提供两组二十轮以上演示：默认配置无降级摘要；用户指定配置触发层1→层2→追加摘要并成功追问初始订单；验证日志和原文仍在、worker不阻塞。浏览器验证新建、切换、回载、继续聊和失败静默。

阶段过程持续追记 dev-notes/ch07.md；计划获批后再实现，完成前运行 verification-before-completion，提交前请求code review，最后给演示命令、实测结果与笔记路径。

## 已查官方接口

Context7 文档：LangGraph graph-api/checkpointers、LangChain `langchain_core.messages.utils.trim_messages` API、SQLAlchemy 2.0 session_transaction/data_update、FastAPI query/path 参数与 response-model。实现前涉及新API继续通过Context7预查；实际安装版本同时核对签名。
