# 第三章：电商客服语义知识库设计

## 目标

将 `query_faq` 的内部实现从 MySQL 关键词 `LIKE` 查询升级为 BGE-M3 dense 向量语义检索，同时保持工具名、输入参数及 JSON 输出结构不变。新增可重跑的知识导入、对话挖掘、去重、MySQL/Milvus 双写与在线检索链路，并提供适用于电商客服的生产质量通用知识语料。

## 已确认的设计决策

- 服务对象是电商系统客服；知识主题覆盖商品咨询、配送与运费、退换货、退款、支付和售后处理。
- BGE-M3 使用用户提供的 SiliconFlow OpenAI-compatible embeddings 服务，模型 ID 为 `BAAI/bge-m3`。探测得到 1024 维向量；endpoint、模型 ID 和 key 从 MewHelp 环境配置读取。
- Milvus 是 dense 向量检索库，MySQL `knowledge_chunks` 是知识原文和状态的权威源。
- 对话挖知识通过 CLI 运行，由系统 cron 调度；应用不增加常驻调度器。
- `query_faq` 的对外契约保持不变；本章不实现关键词召回、混合检索、稀疏检索或重排。
- 通用语料由实现者生成并用于电商客服基线。没有业务资料支持的店铺专属费率、承诺时限、地区例外不编具体数字；回答明确以商品页、订单页、结算页显示或经核实的店铺规则为准。
- Milvus 的主键使用 MySQL chunk 主键；索引写入使用 upsert，Milvus 返回 ID 回填 MySQL。重跑同一 MySQL 主键不会生成重复向量。
- 对话知识先进入 staging，再按规范化后的 category、questions、answer 指纹全局去重入库。
- Markdown chunk 字符数有可配置目标与重叠目标；按句子边界裁切，单句和单表格行若超过目标则完整保留，长度目标不截断事实。

## 范围

包含：

- 标题层级感知的 Markdown 知识解析与切分；超长章节递归切分，重叠只包含完整句子，大表格按行切分并在每块重复表头。
- 面向电商客服的通用知识 Markdown 语料，以及现有 FAQ 的知识化导入。
- 历史 `messages` 的批次抽取任务、checkpoint、问答 staging、全局去重和幂等入库。
- 新增 MySQL `knowledge_chunks` 和问答暂存/抽取游标结构。
- Milvus `knowledge` 集合、BGE-M3 1024 维 dense 向量、pending 补偿和在线 Top-K 检索。
- `query_faq` 保持 `query: str` 输入及命中/未命中 JSON 结构。
- 可由 cron 调度的 CLI、故障恢复演示、FAQ 标注评估集、测试与运行手册。

不包含：

- 关键词、BM25、稀疏向量、混合召回、重排或自动 Agent 循环。
- 未提供的品牌目录、真实 SKU 属性、具体价格、具体运费、退货费用承担规则、赔偿金额或店铺 SLA；通用语料不得将这些内容伪装成已经核实的商家政策。
- 在线文档抓取、PDF/OCR 解析、用户认证/授权改造、生产级分布式调度平台。

## 架构

采用当前 FastAPI/SQLAlchemy 应用内的模块化实现，不拆分额外微服务。离线 CLI 与在线客服共用 MySQL、BGE-M3 embeddings 客户端和 Milvus repository；通过接口注入 embedding 与 vector repository，单元测试使用确定性替身，部署验收使用真实服务。

### 知识字段与文本构造

每个 `knowledge_chunks` 记录包含：

| 字段 | 含义 |
|---|---|
| `category` | 分类；政策类保存父级标题路径，FAQ/问答保存业务分类 |
| `questions` | JSON 问题列表；商品 FAQ 和对话知识写真实问法，政策知识写当前章节标题 |
| `answer` | 原文答案或章节正文 |
| `chapter_path` | 标题路径元数据 |
| `content_type` | `product_faq`、`policy`、`after_sales`、`conversation_qa` 等类型元数据 |
| `is_critical` | 是否关键条款的布尔元数据 |
| `is_active` | 当前来源版本中的知识块标记；旧块保留作清理标记并从在线结果中排除 |
| `previous_chunk_id` / `next_chunk_id` | 同一来源相邻知识块指针 |
| `vector_id` / `vector_status` | Milvus ID 回填及 `pending` / `vectorized` 状态 |

`embedding_text` 固定由 `category`、`questions` 和 `answer` 拼接生成。章节路径、内容类型、关键条款标记和前后块指针只存 MySQL，不进入 embedding 文本或 Milvus 向量字段。MySQL 另保存来源键、内容指纹和必要的失败诊断信息，以支持更新检测与恢复。

### Markdown 切分

解析器按 Markdown 标题级别维护标题栈。每个章节保留完整路径，并按段落、句子边界递归拆分超长内容；重叠从最近的完整句子取得，不产生截断句。中文和英文常用句末标点均作为边界。表格超过块目标时按行拆分，每块重复原始表头和分隔行。代码围栏按完整行/完整围栏优先处理。默认 `max_chars=1200`、`overlap_chars=200`，可由调用配置覆盖；这些是目标而非硬上限，避免拆断单句或表格行。

章节的 `questions` 使用本块当前标题子树中最近的问题标题；离开该问题标题的子树后不继承到兄弟章节。`category` 使用其父级标题路径。无上级标题时使用知识类型的明确根分类。FAQ 记录保留原始真实问句。文档重切分按来源键与块序号稳定生成记录；正文指纹变化时将记录重新标记为 pending，邻接指针在整份来源重建后统一更新。不再生成的旧块标记为 inactive 并从检索结果排除，再删除 Milvus 向量；向量删除失败可由下一次建库继续回收。

### 对话挖掘、暂存和去重

CLI 使用递增的 `Message.id` 游标分批读取历史客服消息，只把用户问题及其后对应的 assistant 最终回答交给抽取模型，跳过 tool 消息、工具调用申请、空回答和无法确认的配对。抽取模型沿用项目现有 OpenAI-compatible 聊天模型配置，要求返回结构化问答候选；解析失败的批次不推进 checkpoint。

每批候选与来源消息 ID 写入 staging 后，才推进该对话的消息游标。游标按 conversation 独立保存，避免全局 Message ID 交错时漏处理其他会话，也避免未回答会话阻塞其他会话。重复执行时 staging/source fingerprint 唯一约束防止重复候选。抽取候选经全局规范化去重后，再与现有知识指纹比较，只有新问答才转成 `knowledge_chunks`。失败批次可从最近已提交游标重跑；去重指纹为 category、问题集合和答案规范化后组合的稳定 hash，不做未经标注验证的语义去重。

### MySQL 与 Milvus 双写和恢复

1. 先在 MySQL 按稳定来源键幂等写入原文和元数据，状态为 `pending` 并提交短事务。
2. 读取 pending 记录，以固定构造文本请求 BGE-M3 embeddings。
3. 将 `{chunk_id, vector}` 以 Milvus upsert 写入 `knowledge` collection，主键字段 `chunk_id` 等于 MySQL 主键。模型向量必须是 1024 维。
4. 读取 Milvus upsert 返回的 ID，在 MySQL 回填 `vector_id` 并将状态设为 `vectorized`。
5. 若 embeddings、Milvus 写入或 MySQL 回填失败，原文保留且状态为 pending；重跑对同一主键执行 upsert，成功后再回填状态。

Milvus 仅持有 chunk 主键与 dense vector；MySQL 始终权威保存 category/questions/answer 和所有要求的元数据。处理过程按有界批次运行，确保单块或单批失败不会回滚已完成知识。

重建 Markdown 来源或导入 FAQ 时，同时协调已删除的知识块：MySQL 先将不再存在的来源块标为 inactive，检索与 pending 同步均排除它们；随后删除旧 Milvus 向量并清空 `vector_id`。删除失败时保留向量 ID，下一次 CLI 运行继续清理。FAQ 仍保留其主键来源键，FAQ 重现时可按同一主键重新激活并 upsert。

legacy FAQ reconciliation 只处理 `faq:` 来源键，避免误清理章节标题虽属于 FAQ 内容、来源仍是 `doc:` 的 Markdown 块。MySQL hydration 只返回 active 且 `vector_status=vectorized` 的行；知识内容进入 pending 的窗口中，旧 Milvus hit 不得映射到更新后的 MySQL 答案。

### 在线检索与工具契约

`build_tools` 获得 FAQ retriever 依赖；LangChain 工具继续名为 `query_faq`，参数仍为 `query: str`。检索步骤是：生成 query embedding、以 COSINE 在 Milvus `knowledge` 集合做 dense Top-5、按命中 ID 到 MySQL 批量取回记录、按 Milvus 排序组装响应。低于配置的 minimum similarity 时不返回不可信结果。

契约：

- 命中：`{"matched": true, "items": [{"question": "...", "answer": "...", "category": "..."}]}`。JSON 字段名、值类型和 `items` 最大条数与现有实现一致；问题字段使用记录的规范问题。
- 未命中：`{"matched": false, "message": "FAQ 未命中关键词：<query>"}`，保持现有字段和文本格式以兼容消费者。
- 向量服务异常按现有 ToolRunner 安全错误边界处理；不把凭据、异常堆栈或内部连接信息返回给用户。

### CLI 与部署

提供三个命令边界：

- `python -m scripts.build_knowledge --source-dir knowledge_docs`：解析 Markdown、导入现有 FAQ、写入 MySQL 并同步 pending vectors。
- `python -m scripts.extract_conversation_knowledge --batch-size 100`：抽取尚未处理的历史消息、暂存、全局去重入库并同步向量。
- `python -m scripts.sync_knowledge_vectors --batch-size 32`：只重试 pending 向量，供故障恢复和 cron 调度。

Milvus 通过 Docker Compose 与现有 MySQL 一起运行。所有新连接参数都从 `.env`/进程环境读取，包括 `MILVUS_URI`、集合名、embedding API base、模型 ID、embedding key、chunk 参数、Top-K、min similarity、批次大小和超时。密钥不得进入源码、README、测试夹具、开发记录或日志。历史消息抽取按用户配置的系统 cron 周期执行；任务结束返回成功/失败退出码并输出处理数，不新增常驻 scheduler。

## 验证

- Markdown 评估：嵌套标题、常规长段落、长单句、含中英文标点段落、大表格、代码围栏；断言章节路径、父分类、完整句重叠、表头复制和相邻块指针正确。
- 对话抽取：验证只取 user/最终 assistant 配对、跳过 tool 内容、批次 checkpoint、抽取异常不推进游标、stage 重跑不重复、跨批次重复候选全局去重。
- 双写恢复：注入 Milvus 写入前故障，确认 MySQL 原文仍 pending；重跑后同一 chunk 主键仅有一个向量、vector_id 已回填且状态为 vectorized。再覆盖 Milvus upsert 成功但 MySQL 状态回填前中断。
- 检索评估：使用带来源 ID 与答案断言的标注集；覆盖“邮费是多少”语义召回运费/结算说明、退换货/配送/退款问题和无关问题。对 min similarity 做固定网格评估，选择保持所有标注正例命中且负例不误命中的阈值；如果语料分布不能同时满足，则报告取舍并停止阈值验收，不能隐藏误召回。
- 工具回归：保持五个业务工具名、`query_faq` 入参 JSON schema、命中/未命中 JSON 字段兼容；确认 API 聊天链路通过原工具结果回答。
- 集成演示：使用 MySQL、Milvus、经验证的 SiliconFlow `BAAI/bge-m3` endpoint 运行构建、对话抽取和断点恢复命令；不在终端打印 key。

## 最新接口依据

- Milvus/PyMilvus 当前 API 用法以 Context7 `/milvus-io/pymilvus` 官方文档检索结果为准：自定义 Int64 primary key 与 FloatVector schema、upsert 返回 IDs、COSINE search 返回 hit IDs/scores/entities。实现前根据选定 PyMilvus 版本再次核对。
- SQLAlchemy 2.0 映射和事务接口参考 Context7 `/websites/sqlalchemy_en_20` 官方文档；项目现有 ORM/`sessionmaker` 模式延续使用。
- SiliconFlow embeddings 官方 API 定义确认 endpoint 为 `POST https://api.siliconflow.cn/v1/embeddings`，模型列表包括 `BAAI/bge-m3`，该模型输入上限 8192 tokens；实际 key 调用返回 1024 维。调用仍通过配置 endpoint/model/key，不在代码固定密钥。
- Context7 与官方 API 文档只作为当前接口依据；若依赖版本与运行环境不同，实施时先按锁定版本校验，不得猜用法。
