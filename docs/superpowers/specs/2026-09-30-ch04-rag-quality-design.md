# ch04：混合检索、重排与评估设计草案

日期：2026-09-30。基线：main `8ec79ef`。状态：用户于 2026-10-01 批准，新增评估集至少 300 条约束。

## 目标与边界

面向现有电商客服，将知识查询升级为 Milvus 原生 BM25 + dense + RRF + bge-reranker-v2-m3；回答可追溯、缺证拒答，四策略评估能输出按 query 类型分桶的真实数字，并持续保存忠实度编造个案。

固定选择：BM25 必须由 Milvus 的 text 字段 BM25 Function 生成；内置 chinese analyzer；双路各 Top-50；hybrid_search + RRFRanker；重排 Top-10。沿用 BGE-M3 1024 维 embeddings、MySQL 原文权威、React/Vite 聊天页。现有 Milvus 3.0 和 PyMilvus 3.x 沿用，实施时核对实际 API。

不做指代消解、多轮 query 改写。同义词只扩展检索输入，不生成多份入库知识。满意度反馈仅采集到前端，不发送后端、不入低置信度池。

## 已确认的兼容决定

用户于本轮答复“接受”：给 conversations 新增唯一 BIGINT UNSIGNED id，保留 conversation_id 字符串主键和现有消息、工单外键；low_confidence_questions.conversation_id 按用户 DDL 关联 conversations.id。

现有会话先由迁移补数值 ID，新会话由 MySQL AUTO_INCREMENT 产生。ORM 使用服务端生成值并回查，不依赖非主键 lastrowid。必须提供显式、可重跑迁移；metadata.create_all 不能替代已有表的 ALTER。新库初始化同样得到兼容结构。SQLite 单元测试采用明确的适配机制；MySQL 实库迁移和新会话自动生成必须独立验证，禁止用 SQLite 通过代替。

## 方案比较与推荐

1. 推荐：在现有 query_faq 工具和 ChatService 内接入独立的 query 理解、检索、生成门禁服务。沿用业务工具路由和持久化；知识回答只从检索证据生成。
2. 全量替换聊天为新 RAG 编排：控制集中，但扩大订单、物流、工单回归范围。
3. 并行新增独立 RAG 接口：隔离简单，但现有聊天页与知识工具会形成两套入口。

选方案 1。无需引入 LangGraph 或 Langfuse；若实现发现现有接口不能满足固定需求，先向用户说明矛盾。

## 入库和集合迁移

沿用 knowledge_chunks 的稳定主键、来源键、正文、questions、chapter_path、category 和 active/vectorized 状态。一个 chunk 对应一份权威原文和一个 Milvus 主键。

新增独立版本集合 knowledge_ch04：chunk_id INT64 主键、embedding FLOAT_VECTOR(1024)、text VARCHAR、bm25 SPARSE_FLOAT_VECTOR，以及可过滤 category/is_active。text 由真实问题与正文组合，保持型号原样；不追加同义改写副本。text 开启 analyzer_params={type: chinese}，挂 FunctionType.BM25；dense 索引 COSINE，sparse 索引 BM25。

从 MySQL active 知识全量生成新集合，保留旧 dense 集合。重建进度独立记录，禁止直接将旧集合 vectorized 状态误当作新集合完成。批量主键 upsert，失败可重跑；完成后核对来源数、主键集合、型号检索和过滤，再通过配置切换。后续更新/删除沿用 pending 恢复，并同步 text 与过滤字段。检索回查只允许 active 且对应版本同步成功的正文，避免旧向量匹配新正文。

## Query 理解与检索

输入为本轮用户原话与可选 category。单轮结构化归一返回 canonical_query 和有限的检索侧同义扩展；不得引用历史，不推断未提供型号或品类。必须保留原话中的型号、数字和否定条件；结构化输出失效时使用原话继续检索并记录降级。

dense 输入归一问法，BM25 输入归一问法与受控同义词，同时保留原始型号。品类过滤白名单化、编码参数并同步应用两条 AnnSearchRequest，使过滤发生在召回前。

四个统一策略接口：dense、bm25、hybrid、hybrid_rerank。dense 与 BM25 各召回 Top-50；hybrid 使用 Milvus hybrid_search 和 RRFRanker 融合取最多 50 候选；hybrid_rerank 使用固定模型对融合候选精排，取 Top-10。其余策略生成时也最多使用前 10 个 chunk，使证据预算可比。

默认使用 SiliconFlow /v1/rerank，模型 BAAI/bge-reranker-v2-m3；base URL、凭据独立可配置，不将其他项目密钥复制到仓库。校验 results.index 唯一、范围有效，relevance_score 有限，映射回原候选。服务失败显式报错，不静默替换模型或检索策略。

RRF/BM25 分数不能使用旧 COSINE 阈值。上线低置信度门槛用标注开发集校准重排分数和证据条件；测试集不参与选阈值。无候选直接拒答；低于门槛拒答并入池。最终阈值及其开发集效果由真实运行报告给出。

## Prompt 排列、引用与生成门禁

每条证据保留 n、chunk_id、source_key、section_path（来自 chapter_path）、question、answer、原文定位。按重排排名分配稳定引用号 [1]…[10]。prompt 布局将第一名放开头、第二名放结尾，其余交替向两端填充；例如排名 1…6 排列为 1、3、5、6、4、2。编号不随摆放次序改变。

知识生成使用结构化输出：sufficient、reason、answer、cited_numbers。模型对原问题及证据全集自评是否足够；知识正文视作数据，不能作为指令。System Prompt 明列不承诺到账时间、未被证据支持的到货时间/退款成功/赔付金额，不编商品规格或售后资格。

结果先缓冲后检查：sufficient=false 则固定明确拒答、source=self_check 入池；充分时检查非空回答、有引用、所有角标均属于当轮证据且与 cited_numbers 一致。不合法输出不向用户发送，按生成自评/校验失败原因拒答并入池。检索低置信度 source=retrieval_low_conf。模型/检索不可用提示服务失败，不伪称知识库无证据。

不调用知识工具却直接回答知识事实会绕过门禁，因此工具路由提示须要求政策/商品知识查询走 query_faq；无工具分支仅允许寒暄和补充信息询问，知识性回答进入受控知识流程。订单/物流/工单工具沿用原路由和结果，回归现有接口。

## 会话持久化、SSE 和来源展示

知识工具内部返回携带完整 evidence 的结构，原 matched/items 基础字段可保留；在线不受旧 Top-5 限制。服务级传递证据，不把截断工具文案当成来源快照。

assistant Message 新增可空 citations JSON，保存当轮全部喂给模型的证据快照。SSE 增加 citations 事件，包含引用号、原文、章节路径、来源定位；token/done/error 保持。证据只绑定当前回答，不串入上一轮。

前端点击有效 [n] 打开来源面板，显示当轮 chunk 原文与章节路径，可跳转到来源文档章节。Markdown 文档由受限来源路由提供，路径只能解析已登记 knowledge_docs 内来源，拒绝穿越；锚点生成与 chunk 定位规则一致。会话抽取来源显示脱敏 Q&A 与可用来源标识，不公开其他用户原始会话。证据历史快照与当前原文更新后可不同，界面明确展示当轮证据。

前端每条已完成的真实 assistant 回答左下角显示 👍/👎，点击即点亮、显示“已反馈”、锁定两按钮；消息级状态及本地存储记录 conversation_id/message_id/choice/time，重渲染和重复点击不重复采集。欢迎语和未完成/连接错误消息不采集。使用 Vibe Coding，做构建及交互验收，按用户要求免 brainstorm/TDD/code review。

## 两张台账

low_confidence_questions 按用户 DDL 建表，保留原话、数值会话关联、source、reason、created_at。检索拒答和生成不足均在发出成功 done 前持久化，入池失败报告错误，不声称已完整处理。source=user_feedback 保留给后续，本章前端反馈不使用它。

faith_cases 按用户 DDL 建表。一题一行，四策略诊断结果全部保存在报告，持久台账默认收录 hybrid_rerank 的编造个案，避免同一轮多策略相互覆盖并误增次数。重复 eval_id 更新最新 query/bucket/strategy/answer/reason/完整 citations/judge_model，seen_count 加一、last_seen_at 更新、first_seen_at 不变。

再次判编造时，已处置状态回到“未解决”，清空 resolution，保留 resolved_at 作为复发痕迹。本轮未判编造不自动解决历史个案。提供查询和状态处置 CLI；标“已解决”或“无需解决”必须填写非空且不超过 300 字说明，更新时间；退回未解决清空说明并保留处置时间。不扩展独立管理前端。

## 评估设计

建立固定版本电商评估集：A_policy 政策、B_model 型号、C_colloquial 口语、D_unknown 无知识、E_multi 跨片证据。至少 300 题（每桶 60 题，五桶共 300；每桶校准 12 题、测试 48 题），easy/medium/hard 梯度，区分 calibration 与 test；记录 eval_id/query/category/难度/ground-truth 来源键和章节/必要答案事实/应否拒答。来源标签使用稳定 source_key，建库后解析 chunk_id；标签失效直接报错。

型号语料标为受控测试商品说明，供隔离验收语料使用，绝不编造真实商品规格混入生产知识。真实客服语料和评估快照一起版本化，测试与线上集合分开配置。

四策略共用同一评估集、query 归一、过滤、生成模型、证据预算和裁判，分别记录阶段延迟、候选 ID、最终 Top-10、答案、证据全集、自评与引用校验。检索报 Recall@1/5/10/50 和 MRR；重排最终列表只报其实际支持的 K（至 10），同时独立报候选 Recall@50。多个相关块时 Recall 为命中相关块数/全部相关块数；MRR 为第一相关块倒数排名；D_unknown 不纳入检索正例分母，单列拒答率和误答率。

Faithfulness：裁判逐个可核查事实声明判断证据支持比例，保留具体不支持句和理由；无事实声明（包括拒答）记 N/A，单列应答题误拒答率、未知题拒答率及空答率，避免用全拒答刷忠实度。裁判输入只有原问题、实际答案、当轮证据全集，不能偷看 ground-truth 来支持答案。裁判解析或调用失败标 error，不计为通过。

输出带 run_id 的 JSON 和 Markdown，不覆盖历史，包含配置/模型/语料摘要、总体与 bucket/难度指标及逐题明细。提供离线 fixture 模式验证公式/持久化和完整管道接线，但标注 synthetic，不充当真实 Milvus/模型效果；验收数字来自 live 四策略运行。

## 验证和阶段留痕

后端实施先计划再 TDD；纯 prompt/语料用标注样例验证替代单测。新增 MySQL 会话 ID 迁移/插入、表约束/复发语义、BM25 schema、召回过滤、重排索引映射、布局编号、门禁绕过、SSE 来源归属、引用合法性、指标计算和裁判失败用例。

实库验收必须证明：四策略分桶报告有真实数字；带型号问法 BM25 命中标注来源；聊天角标点开原文并定位章节；未知问题明确拒答且 low_confidence_questions 包含原话与会话；faith_cases 跨轮计数和复发符合 DDL。完整 Python 回归、前端 build、compose config 和 diff 检查在最终交付前重跑。

dev-notes/ch04.md 每阶段即时追加关键原话、产出路径/评审、用户纠偏、失败与返工。最终交付演示命令、测试结果和记录路径；不把尚未真实跑过的验收描述为通过。

## 文档核查记录

Context7 于 2026-09-30 查询：/milvus-io/milvus-docs（BM25 Function、hybrid_search/RRFRanker、chinese analyzer）、/websites/siliconflow_cn（/v1/rerank 与 BAAI/bge-reranker-v2-m3）、/websites/sqlalchemy_en_20（MySQL unsigned BIGINT、服务端生成值）。SQLAlchemy 检索返回的非主键 IDENTITY 示例属于 MSSQL，不作为 MySQL 用法依据；实施需显式 MySQL DDL 并实库验证。

每项涉及 FastAPI/SQLAlchemy/LangChain/PyMilvus 的实现前继续通过 Context7 核对对应接口及安装版本。凭据只本地配置；重排模型不可用或其他固定技术冲突时暂停询问用户。
