# ch04 评估区分度增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Native 执行沿用本轮已有方式；完成后按 requesting-code-review 做一次独立评审。

**Goal:** 保留 v1 回归基线，新增至少 300 道更有区分度、可追溯标注的 v2 题目，真实比较四种检索策略，区分召回、答案正确性、忠实度与拒答质量。

**Architecture:** 沿用现有评估脚本，增加显式数据集入口和版本隔离。v2 使用更大的受控语料、近似项和条件干扰项，标注冻结后统一运行四策略；生成模型不变，裁判身份据实披露。

**Tech Stack:** 现有 Python、Milvus 原生 BM25/chinese analyzer、hybrid_search/RRF、BAAI/bge-reranker-v2-m3、LangChain ChatOpenAI；不引入 Luna，不修改前端。

**Spec:** 既有 `docs/superpowers/specs/2026-09-30-ch04-rag-quality-design.md` 及下述用户已接受的局部设计。

## 已接受的局部设计

用户关键原话：“现在几种方式的成功率是否都太高了,想办法做出区分度”；在澄清出题者、答案生成者、裁判三种角色后，用户确认“按照你的建议来”。

v1 300 题来自 100 个 topic，只涉及 40 个 gold chunk；隔离语料约 60 chunk，240 道正例中 180 道仅需一个 chunk，category 全部为空。这些是改进依据，不能将旧 Recall@10 解读成答案成功率。

新增 `evaluation/ch04/v2/`，旧数据和报告不覆盖。v2 至少 300 题，五桶各 60、每桶三档难度各 20，60 calibration/240 test。每题标注来源、必要事实、应否拒答、全局语义 family、挑战标签和干扰来源。校准和测试按全局 family 分离，同一主题不得通过换桶或改写流入另一 split。

语料目标不少于 400 个有效 chunk，来自有实际区别的商品规格、配件兼容、品类政策、活动适用条件等受控场景，不靠重复文本凑数。虚构资料仅入隔离评估库。不存在统一的“更难所以所有策略都必须下降”验收线，也不预设混合重排一定最佳。

## Global Constraints

- 固定召回 dense/BM25 各 Top-50、Milvus RRF、固定重排 Top-10，不调技术选型以迎合题库。
- 当前客服生成模型不变，不引入 Luna，也不把 Codex 工作模型名称直接填入模型服务端点。
- 本地没有 JUDGE_MODEL/JUDGE_API_BASE/JUDGE_API_KEY 独立配置。不得声称已接入独立裁判；凭据和端点不能从猜测中生成。
- 先完成题库增强。已有默认裁判可继续作为带明确限制的基线；未来接入独立裁判后可以对保留答案重判，不必重新调用生成模型。
- 新题号使用 V2A001 等，不与 v1 的 faith_cases.eval_id 冲突。一题一行的原台账语义不变。
- v2 SQLite/Milvus 与 v1、生产集合隔离；每轮报告记录题库/语料 SHA256、模型与阈值。
- 只在 calibration 上分析阈值；首次对比保留现有 0.05，避免在 test 上调参。若另做校准实验，作为单独报告保留。
- 题库和 Prompt 以标注样例验证替代 TDD；评估脚本和指标变更先写失败测试。
- 涉及现有库接口变更前用 Context7 核对；2026-10-02 已查 LangChain Reference 的 Pydantic/json_mode/invoke 约定，正式实现时核对安装版本。
- 每阶段完成即追加 dev-notes/ch04.md；不合并、不推送。

## Review Focus

- 更换数据集后仍误用 v1 corpus/集合：用两个不重叠的小数据集验证加载及默认数据库/集合隔离。
- 同一政策换表达或换桶泄漏到 test：负例让不同桶共用 family_id 且跨 split，验证明确拒绝。
- category 过滤题 gold 不在指定类别：验证来源存在且类别一致，不能用不可命中的错误标注制造低分。
- 单条命中多证据题就被判成功：用两条 gold、只检到一条的样例，验证 Recall 为 0.5、完整证据命中为 0。
- 拒答、裁判失败、证据支持但答非所问被算作正确：分别检查答案覆盖和忠实度，裁判失败记错误，拒答正例的正确性为 0。

---

### Task 1: 可复现的数据集入口和版本隔离

**Files:** 修改 `scripts/evaluate_ch04.py`、`scripts/validate_ch04_dataset.py`；新增 `tests/test_quality_dataset_v2.py`，扩展 `tests/test_quality_evaluation.py`。

**Interfaces:** `load_corpus(root: Path)` 保持现有接口；新增 `--dataset-dir`，默认 `evaluation/ch04`，读取其 cases.json/corpus。`build_live(dataset_dir, drafts)` 使用同一批已验证 drafts；v2 缺省隔离 SQLite 为其 eval.db、Milvus 为 knowledge_ch04_eval_v2。显式 EVAL 配置的覆盖行为记录在元数据。

- [x] RED：建立两个临时目录，各含不同的 cases.json/corpus，验证指定目录的来源进入报告，缺文件明确报错，v1 默认不变；测试不连接外部服务。
- [x] GREEN：参数贯穿加载、校验、建库、索引、SHA256、默认报告目录；禁止只替换 questions 而保留旧 corpus。
- [x] 跑 dataset/evaluation 专项回归；记录 Task 1 产出和 RED/GREEN，提交本任务。

### Task 2: v2 语料、300 题及冻结标注

**Files:** 新增 `evaluation/ch04/v2/{cases.json,README.md,manifest.json,annotation-audit.md,corpus/*.md}`；扩展 validator 和相应测试。

**Interfaces:** 保持既有题目字段；v2 增加 `family_id`、`challenge_tags`、`distractor_source_keys`。manifest 记录 schema_version、数据分配、题库/语料 hash、冻结时间、作者和复核限制。

- [x] 数据验证替代 TDD：先写 15 个桶×难度单元的标注样例，检查每个 required_fact 可由 gold 原文支持，拒答理由为缺失事实而非猜测事实为假。
- [x] 生成至少 400 个有效 chunk：精确型号与后缀、同名不同品类、不同时间/地域/购买渠道/状态的规则，全文内有明确适用条件；同义词只写入查询，不为一个事实拆存多个同义版本。
- [x] 完成 300 个不同 query：五桶各 60；至少 30 个实际 category 过滤题；E_multi 至少含两条必要证据，并包含三条或以上的样例；D_unknown 含完全未知、相关但缺关键值、错误前提三类。
- [x] 全局 family 按 60/240 分离；型号查询避开规则递增值与简单奇偶规律，防止凭题号推断答案；标注不引用干扰 chunk。
- [x] RED→GREEN：validator 拒绝跨桶 family 泄漏、gold/category 不一致、gold/distractor 重叠、缺少必要事实、题号冲突；v1 校验继续通过。
- [x] 逐题静态核对标注来源与事实；人工式抽查 60 题（每桶每档至少 4 题），将出处、条件、拒答依据逐条写入 annotation-audit.md。披露这是助手核验，不冒充业务专家复核。
- [x] 冻结题库和语料 hash，运行 validator；追加阶段留痕并提交。之后不得根据 test 分数删题、换题或改 gold；发现标注错误必须版本化并记录。

### Task 3: 区分召回、完整性、答案正确性与忠实度

**Files:** 修改 `app/services/quality/evaluation.py`、`scripts/evaluate_ch04.py`；扩展 `tests/test_quality_evaluation.py`。

**Interfaces:** 保留 FaithfulnessJudge 只看 query/answer/当轮证据；新增独立的答案覆盖判定函数，只向该判定传 query、answer、required_facts、should_refuse。两个判定分开输入，ground-truth 不泄漏到生成和忠实度裁判。

- [x] RED：手算两 gold 只检中一个的 `all_evidence@5/10` 为 0，全部检中为 1，未知题为 N/A；重复命中不提高分数。

```python
def test_multichunk_completeness_is_not_first_hit():
    partial = retrieval_metrics(['a', 'noise'], {'a', 'b'})
    complete = retrieval_metrics(['a', 'b'], {'a', 'b'})
    assert partial['recall@5'] == 0.5
    assert partial['all_evidence@5'] == 0
    assert complete['all_evidence@5'] == 1
    assert retrieval_metrics(['noise'], set())['all_evidence@5'] is None
```

- [x] GREEN：扩展 retrieval_metrics 并汇总，报告保留 Recall@1/5/10、MRR、候选 Recall@50、完整证据命中率及分母。
- [x] RED：证据支持的答案可能漏掉必要事实；正确拒答未知题、误拒答正例、裁判解析失败分别有不同结果；FaithfulnessJudge 的输入仍不含 required_facts。
- [x] GREEN：新增结构化逐必要事实覆盖/矛盾判定，校验 fact 索引完整唯一。正例整体正确要求全部必要事实覆盖且无矛盾；正例拒答计 0；未知题正确拒答计 1。错误记 null+error，不伪装为通过；报告同时列正确性样本数、错误数与拒答率。
- [x] 报告按桶、难度、challenge_tags 和 test 分组，使用同题配对输出四策略差异及胜/平/负计数，不仅展示单个总平均。
- [x] 元数据/Markdown 根据实际 generator_model、judge_model、端点身份判断是否共享裁判；不把配置字段存在当作独立性证明。不展示密钥。
- [x] RED→GREEN：新增 `--judge-only <rows.jsonl>` 互斥于 live/fixture，逐行读取已有 query/answer/citations/required_facts/refused 和检索指标，保留 generator 身份，不调用 retriever/generator。测试用会抛异常的 retriever/generator 确认重判不触发它们；坏 JSON 行明确拒绝，拒答行忠实度仍 N/A。重判写全新 run，输入日志保持字节不变。
- [x] 新增 `--require-independent-judge`：比较规范化 API 端点与模型身份，相同组合在启动前明确报错；缺配置也报错，不静默复用原模型。非独立基线允许显式沿用当前默认裁判，报告身份属实。不同名称仍只说明模型配置不同，不宣称训练来源完全独立。
- [x] 专项回归通过，记录局限及新指标定义，提交本任务。

### Task 4: 复核、真实对比和交付

**Files:** 新增 `evaluation/ch04/v2/runs/<run_id>/*`；更新 v2 README、仓库 README、dev-notes。

- [x] verification-before-completion：完整 pytest、validator、fixture CLI、git diff --check。fixture 仅验接线。
- [x] requesting-code-review：一次独立评审本轮评估改动和分层标注样例；若没有业务专家参与，报告明确保留此限制。重要问题按 receiving-code-review 修复；代码缺陷先 RED 再 GREEN，数据错误保留修改记录。
- [x] 先对 15 个代表样例做真实 smoke，确认目录/集合/品类过滤/引用/裁判输入可追溯；不依 smoke 分数调整 test 题目。
- [x] 真跑 `python3 -m scripts.evaluate_ch04 --live --dataset-dir evaluation/ch04/v2 --split all --workers 8`，300×4=1200 结果，逐题保存证据、生成模型、裁判模型、判定和错误。新答案正确性判定增加一次裁判调用，耗时单列。
- [x] 如外部调用失败，保留错误和原 run；修复原因后新建 run，不覆盖历史、不将失败算正确。若仍使用同模型裁判，以“同模型裁判基线”标注，而非独立裁判验收。
- [x] 报告解释各策略在哪些类型胜出、召回失误与生成失误分别是什么；不承诺某策略必胜。列 v1/v2 语料规模差异，不将两个难度不同的数据集分数直接解释为回归退化。
- [x] 验证可重判入口：通过 smoke 的 rows.jsonl 重判，不重新生成；独立 JUDGE 配置缺失时明确拒绝“独立裁判”模式，保留已有基线。独立服务具体模型待有配置时确认，不自行启用新供应商或猜型号。
- [x] finish：交付命令、测试结果、报告和 dev-notes 路径；当前分支保留，等待原有集成选择。

## 计划自检

题库区分度、旧版可复现、过滤覆盖、全局分组防泄漏、拒答与必要事实正确性、实际裁判身份披露均有任务和验证落点。没有新增检索技术或前端工作。现阶段可完成题库/指标/同模型基线；独立裁判实跑需要有效的独立服务配置，不能把尚未配置的裁判包装成已完成。
