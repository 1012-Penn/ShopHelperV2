# ch04/v2 针对性诊断

日期：2026-10-02。针对冻结运行 `20261002-120853-23796104` 做了三组小规模诊断；未重跑 1200 组评估，未改生产代码、阈值或题库，成果保留在当前开发分支。凭据只从现有配置读取，没有写入证据。

模型沿用 `deepseek-chat`、`BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3`。总计 25 个外部请求：归一 4、embedding 7、rerank 2、生成 12；另有 21 次只读 Milvus top-50 搜索。重试设为 0 以约束请求数；这不同于生产配置的自动重试。原始响应和输入均分文件保留。

## 已证实

- **归一回退：** 历史四策略各 300/300 都回退到原 query。300 道输入都含 ASCII 标识，100 道命中中文数词正则，140 道命中当前否定字符集合，否定分支无法单独解释全量回退。4 条新请求都删掉了型号/金额标识：A003 删除 `MX-474`、`EV-687`、`220`；B001 删除型号；C001 删除 `MX-474`；C003 删除 `MX-474` 和“同一段”的“一”。现有守卫对这些候选回退是正确的。提示词要求不新增标识且保留否定条件，但未明确要求逐项保留原标识和数字。离线等义反例证实守卫另有边界：`分别` 的“别”触发否定精确匹配，`一点` 被数词规则捕获，重复型号简写会触发多重集不等。反例与生产 `QueryNormalizer` 输出一致。
- **检索路由：** 七题每题使用历史 canonical/lexical/category，查询 dense@50、BM25@50、hybrid@50。官方 PyMilvus `RRFRanker(k=60)` 公式对当前列表复算后，hybrid 50/50 的候选 ID 集合吻合、分数匹配且无严格分数逆序。接口 gold 在 E012/E021/E051/E060 的 dense 排名为 45/43/41/48，不在 BM25 前 50；RRF 严格排名为 56/53/51/61，低于融合截断。E057 未进入任一单路 top-50。E048/E054 当前 hybrid 排名 37/40，full rerank@50 排名 19/26，故落在 rerank@10 之外。C001 的历史 BM25 gold 在候选第 15 名，最终 top-10 将其截掉。
- **协议重放：** B021、B036、C022、C034 各用原 query、历史证据、原 prompt/schema 重放 3 次。12/12 都解析成功且通过生成门禁；引用编号集合与 `cited_numbers` 一致。温度为 0 时多轮输出仍有差异。

## 未复现与限制

历史没有保存四条协议失败的原始生成响应，12 次同输入重放均未复现，不能确定原始失败究竟是引用缺失、越界、编号不一致，还是 JSON 解析错误。E057 只能定位为两路 top-50 均未命中，50 名以后位置未知。检索复测不是历史精确重放：E051/E054 当前 hybrid ID 集合相同，但分别有 5/14 个位置变化；历史完整 rerank 排名也未保存。新重排位置只能说明当前排序。

## 输入、工具查阅与复跑

`cases.json` SHA-256：`6bf2e08bbf06f5a8f775dca841e88660ec93ae21efc9619323b767b46e5403e6`；历史 `rows.jsonl`：`334928ca5ca36b94ae696e90fd7f20de17a0eca5db457291ab464709096ca8fc`；`eval.db`：`9bcfe606e1df7b89ab36b99ede2de88ba934177739c0bf78b95860ac64f5039b`。冻结 corpus 与 SQLite 480/480 一致（主智能体独立复核）；详情见 [review-verification.json](</root/.codex/worktrees/ch04-rag-quality/MewHelp/evaluation/ch04/v2/diagnostics/20261002-luna/review-verification.json>)。

Context7 官方文档确认 LangChain `include_raw=True` 返回 raw/parsed/parsing_error、OpenAI SDK `max_retries=0` 关闭重试，以及 PyMilvus RRF 默认 `k=60`：[LangChain](https://docs.langchain.com/oss/python/langchain/models)、[OpenAI Python SDK](https://github.com/openai/openai-python/blob/main/README.md)、[PyMilvus](https://github.com/milvus-io/pymilvus/blob/master/_autodocs/api-reference/rankers.md)。

```bash
python3 evaluation/ch04/v2/diagnostics/20261002-luna/probe.py --stage offline
python3 evaluation/ch04/v2/diagnostics/20261002-luna/probe.py --stage normalize
python3 evaluation/ch04/v2/diagnostics/20261002-luna/probe.py --stage route
python3 evaluation/ch04/v2/diagnostics/20261002-luna/probe.py --stage rrf-audit
python3 evaluation/ch04/v2/diagnostics/20261002-luna/probe.py --stage generation --iteration 1  # repeat with 2 and 3
```

`offline` 与 `rrf-audit` 不访问模型；其余阶段会发出上述请求。验证已执行 `py_compile`、只读输入/hash核对、synthetic guard parity、RRF top-50 parity，以及主智能体独立复核。脚本与 raw/parsed JSON 证据均在 [diagnostics/20261002-luna](</root/.codex/worktrees/ch04-rag-quality/MewHelp/evaluation/ch04/v2/diagnostics/20261002-luna>)。
