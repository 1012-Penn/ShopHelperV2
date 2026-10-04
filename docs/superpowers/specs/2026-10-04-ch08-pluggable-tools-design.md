# 第 8 章：即插即用工具系统设计提案

日期：2026-10-04。状态：用户以“执行”确认设计并授权推进；用户以“行,开始执行”确认实施计划；完成证据见 dev-notes/ch08.md。基线核查补充：本章计划从本地 codex/ch07-integration（4eac38d）创建独立工作分支，保留第 6、7 章能力。

## 目标与已确认约束

内置与 MCP 工具进入同一注册表，主力 Agent 每轮拿到最新可调用清单；执行引擎集中执行 JSON Schema 校验、本地权限、确认控制、超时、有限重试、结果格式化和审计。新增已注册内置工具不改 Agent 核心；MCP Server 增加工具只重启对应 Server，客服无需重启。

用户已确认：MCP 未知工具默认拒绝执行；新增工具允许同步修改可热加载的本地权限配置。工具用途描述及 Server annotations 均不是授权依据。唯一获准写工具是内置 create_ticket。

技术选型固定：官方 MCP Python SDK、Streamable HTTP、langchain-mcp-adapters MultiServerMCPClient、现有 LangGraph/SQLite checkpointer、FastAPI、SQLAlchemy。具体 API 使用前查 Context7 并核实安装版本；出现兼容性阻碍时询问用户，不换库。纯 Prompt 以标注样例真实评估；前端卡片按用户指定 Vibe Coding，不套 brainstorm/TDD/code review。其他后端走 Superpowers/TDD。

## 现状与方案比较

当前 app/tools/registry.py 只有固定字典、Pydantic 校验和线程超时；参数错误抛异常，普通异常被一律重试，没有审计。主力 Agent 在 app/services/workflow/agent.py 写死 READ_ONLY 并每次重建工具。内置 query_logistics 与 create_ticket 位于 business.py。

正式首页是 app/static/index.html。投诉按钮经独立 POST /api/v1/tickets 建单，已有消息归属校验、提交预占与重复点击保护；需保留产品行为。当前 main 工作树没有第 6 章订单选择器，但本地 codex/ch07-integration 已含 _offer_orders/_await_order 与 Command(resume=...) 以及第 7 章 ContextManager。本章以该集成版本为实施基线，复用其会话锁、归属/陈旧选择校验、独立展示/等待节点模式与 SqliteSaver，并给工单新增独立确认接口。

方案 A（推荐）：独立 ToolRegistry、ToolPolicy、ToolEngine、MCP discovery 模块，图节点只负责决策编排和确认。职责清楚，权限和审计可单测，新增工具无需改 Agent。

方案 B：直接使用 LangGraph ToolNode 与 adapters hooks。能减少调用包装，但现有同步图、持久化工具消息、统一业务分诊和审计仍需较多额外控制，容易出现两处权限判断。

方案 C：只扩展旧 ToolRunner。改动小，但把动态发现、权限、确认和执行混在一处，后续新增 Server 更难验证。推荐 A，保留兼容适配以避免破坏旧服务入口。

## 注册与动态发现

工具记录至少包含 name、description、JSON Schema、source（builtin 或 mcp:<server>）、执行器；本地附加执行权限、超时与格式化策略。注册接口验证三项必备信息、Schema 合法性及工具名冲突，冲突拒绝登记，不能覆盖内置写工具。

内置工具在服务初始化登记，通过执行上下文注入 conversation_id/session_factory，避免把会话绑定进全局共享工具。运行期 register() 生效，无须重启。内置四项为 query_order、query_product、query_faq、create_ticket；query_logistics 从内置目录移除，只由物流 MCP 提供。

MCP 配置两个 Server，逐个调用 MultiServerMCPClient.get_tools(server_name=...) 以记录可靠来源；每轮 Agent 决策前重新发现，按 Server 原子替换其工具子集。动态工具 Schema 一并更新，不复用永久缓存。单个 Server 失联不拖垮另一个 Server或内置工具；发现有超时，状态明确记录，不能把旧数据伪装成新查询结果。

模型看到已获本地授权的工具，执行时再次检查最新策略，防止绑定后授权被撤销。未知或被撤销工具的实际调用仍返回权限拒绝并落审计。策略配置变更非法时对受影响外部工具关闭授权，不能保留旧授权悄悄放行。绑定最新清单时继续经第 7 章 convert_to_openai_tool、ContextManager.check 与预算日志；高风险政策分支保留原知识/订单闸，明确建单请求在独立建单分支处理，不能借工具升级绕过退款资格校验。

## 权限与输入校验

本地热加载配置以 (source, tool_name) 精确授权，只读与写分开；两个 Server 的初始业务查询显式列入只读许可，外部写工具默认拒绝，即使自称 readOnly 也无效。新增工具注册不等于获得外部权限。

每次调用统一按 JSON Schema 原始参数校验，不能把字符串数字自动转成数字后假称校验成功。必填、类型、enum、范围、空描述与额外字段遵循 Schema；错误包含字段路径和可理解原因，转为 ToolMessage 回灌模型。校验不调用工具，不重试。格式不合法的调用参数也要产生一次审计。

create_ticket 的描述、类型均必填且非空，问题描述不得由模型编造；Prompt 样例覆盖缺描述主动追问。执行引擎不读取模型给的 confirmed/permission 参数作为许可，只接受后端依据会话状态签发的确认上下文。用户明确建单诉求记在服务器会话状态中，跨补充信息轮次保留，在取消/完成后清除；模型自身的工具选择不能产生授权。明确诉求识别采用保守规则及标注负例（否定、引用、仅咨询、第三方工具数据指令）；识别不明确时追问，前端确认仍为必需。

该意愿存于独立 TicketIntent 表（conversation_id、user_id、operation_id、status、原始明确请求及时间），不会因新一轮重建图 initial state 丢失。读取前验证归属，否定撤销、取消、完成或结果未知时关闭该意愿，禁止后续普通消息继承已用许可。后端确认上下文仅由已验证的 checkpoint/意愿生成；confirmed_call_id 必须与本次 tool_call_id 相等，提交收据原子预占后只允许一次从 reserved 到 running。

## 工单预览与恢复

调用数据先进入可 checkpoint 的准备节点；缺参或无明确建单意愿直接走引擎拦截，向模型返回错误；有效 create_ticket 进入独立 ticket_confirmation 节点，调用 interrupt() 输出工单类型、问题描述、tool_call_id 与关联确认标识。

SSE 发送 ticket_preview，正式聊天页显示预览卡片及“确认提交”“取消”。挂起结束当前 HTTP 流并释放会话锁，不持锁等待用户。新增 resume 请求携 conversation_id、user_id、确认标识、approve；后端验证归属与当前 checkpoint 中待确认调用，参数仍取服务端保存的预览，不接受浏览器改写参数。

确认节点自身没有工具调用、数据库写入或审计副作用；恢复通过 Command(resume=...)，后续独立执行节点调用引擎。已处理的只读调用位于确认节点之前的独立 checkpoint，不能因确认节点重跑而重复调用。取消也进入引擎拒绝分支，create_ticket 落 permission_denied 审计并回灌模型，不落 tickets。

同会话有待确认卡片时普通新消息明确提示先确认/取消，不能静默覆盖挂起状态。resume 重复点击、旧卡片或跨用户请求不能产生第二次写入；以会话＋tool_call_id 保留服务端提交预占和结果。写入超时返回“提交结果暂未确认，请勿重复提交”，禁止重新执行同一请求。断流后可查询待确认状态并重新显示卡片。

投诉分支仍固定安抚及按钮建议；原 POST /api/v1/tickets 的用户按钮确认、消息校验和预占不变，只向统一引擎传服务器确认上下文，不额外弹聊天预览。聊天明确建单意愿优先进入 Agent 以追问/预览；没有明确建单诉求的投诉仍走原流程。

两条入口各以服务器生成的请求身份防重复：投诉按钮沿用消息 ID 对应的请求，聊天沿用 operation_id 和工具调用 ID。同一请求/卡片的重复或并发确认只执行一次；用户后来另行明确要求聊天建单属于新的请求，可独立建一单，不按描述相同自动合并。旧投诉卡片仍按原产品行为可确认，不在本章新增跨入口自动关闭。

## 执行、结果与审计

所有在线调用统一引擎入口，执行上下文含会话、调用 ID、来源和服务端确认。错误三类：invalid_arguments、not_found、execution_error；权限独立标记。not_found 是正常空结果，不重试。只读工具仅对 TimeoutError、连接中断等明确暂时故障重试，普通业务异常不重试；写工具不自动重试，包括超时。计数为额外尝试次数，默认写始终为 0。

同步内置工具用有界线程池，超时无法强停线程，明确记录可能仍在后台运行；MCP 异步调用用有界等待并取消会话。结果将 status 等内部枚举按工具格式化策略转中文，按 Schema/格式化器选取回答所需字段；新工具可注册自身结果投影，默认保留业务返回并去除协议包装及内部元数据，不能任意截断关键事实。JSON ensure_ascii=False。MCP isError 不是成功结果，不能误报；网络与业务错误保持分诊。

新增 tool_audit_logs 表，不挂外键：id、conversation_id、tool_call_id、tool_name、source、arguments(JSON)、result_summary、status、error、retry_count、duration_ms、created_at。状态为 success/failed/timeout/validation_blocked/permission_denied，界面或说明映射对应中文。所有执行、拦截和取消均记录；一次逻辑调用一条终态记录，重试不逐次加终态行。审计用独立事务，失败仅记录服务日志，不改变业务结果或阻止执行。待确认阶段不假记成功，最终确认/取消才记录终态。

## 两个独立 MCP 进程

物流 Server 提供 query_logistics(order_id)：承运商、当前状态、随机轨迹、模拟数据标记。售后 Server 提供 query_warranty(order_id)、query_return_progress(order_id)：在保状态/期限或退货处理进度及模拟标记。内部随机 mock，均无数据库、无真实外部系统。独立模块命令启动，默认不同端口与 /mcp endpoint；SDK 主版本需与 adapters 兼容，核实后锁定兼容依赖范围。

## 验证与交付

后端红绿测试：Schema 类型/必填/范围；未确认与外部伪装只读拒绝；超时重试与业务空结果不重试；写超时零重试；审计失败不阻断；动态注册；工具冲突；撤销权限；恢复重复/旧确认/用户归属；投诉按钮回归。真实两进程 HTTP 集成测试覆盖发现、查询、单 Server 重启后新增工具及本地权限热更新。

真实模型标注集覆盖明确建单缺描述→追问→用户补充→预览→确认/取消、否定建单、仅咨询、物流轨迹/在保/退货查询和新工具路由；fixture 只能验证接线，不能替代模型表现。前端 Vibe Coding 后用页面实际操作/浏览器验收确认与取消效果，不强行套前端 TDD/review。

演示脚本覆盖六条用户验收，输出 tickets 与审计查询；迁移脚本支持已有数据库。最终交付启动/演示命令、实际测试及评估结果、dev-notes/ch08.md 路径。阶段记录当时追加，包含原话、产出、纠偏、翻车返工。

本章不实现 Skill、仓储系统、真实业务集成或多副本 checkpoint。现有未提交的其他章节文件保持原样。
