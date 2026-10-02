# ch04 hybrid fusion output-limit calibration

状态：`complete`；split：calibration 60；test：0。

本实验只改变融合输出 `limit`（50 / 100）。Milvus dense 与 native BM25 每路仍 limit=50，RRF、`k=60`、最终重排 top_n=10、reranker 与 0.05 阈值均不变。未知题的 gold 检索指标为 N/A。延迟只计 embedding、两次 hybrid recall、hydration 和 rerank，不含改写、生成、裁判或 normalization。

| 融合输出上限 | 有效 gold 题 | Recall@10 | 完整证据@10 | MRR | 候选 gold Recall | 候选完整 gold | 平均候选数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 48 | 0.993 | 0.979 | 0.906 | 0.993 | 0.979 | 50.000 |
| 100 | 48 | 1.000 | 1.000 | 0.906 | 1.000 | 1.000 | 68.267 |

## 平均阶段耗时（秒）

| 上限 | embedding | recall | hydration | rerank |
|---:|---:|---:|---:|---:|
| 50 | 0.159 | 0.025 | 0.001 | 0.613 |
| 100 | 0.159 | 0.014 | 0.001 | 0.805 |

## 请求上限

最大 embedding 调用 60、Milvus hybrid_search 调用 120、rerank 调用 120；每题一个向量供两个融合查询复用，两个融合结果各至多重排一次。没有改写、生成或裁判请求；异常即停止，脚本不重试。SQLite 使用只读 URI；不创建表、索引、集合或账本。

输入及逐题完整候选、重排结果、gold ID/source key、分段耗时和 hashes 见 `results.json`。

## API 与实现依据

PyMilvus 文档分别定义了每个 `AnnSearchRequest.limit` 和融合 `hybrid_search(limit=...)`，并记录 `RRFRanker(k=60)` 默认值：[AnnSearchRequest / RRFRanker](https://github.com/milvus-io/pymilvus/blob/master/_autodocs/api-reference/rankers.md)、[hybrid_search](https://github.com/milvus-io/pymilvus/blob/master/_autodocs/api-reference/async-milvus-client.md)。SiliconFlow 定义 `/rerank` 的 `top_n`：[Create Rerank](https://docs.siliconflow.com/en/api-reference/rerank/create-rerank)。OpenAI SDK 的 `max_retries=0` 关闭自动重试：[OpenAI Python SDK](https://github.com/openai/openai-python/blob/main/README.md)。

运行时实现文件 SHA-256：
- `app/services/knowledge/hybrid_store.py`: `a7ff5a275c292d23b21bf03f99b0a60478eadef8916825f634dfd5d0f02c0bf8`
- `app/services/quality/retrieval.py`: `eb40544a0fd4f782791cd5f8be84e9d4d31ce8f16f4ce841f6623a7ad5f45cfc`
- `app/services/quality/rerank.py`: `bfe9c26ba917c420f89791801bec4360208ad8a5ec9d5ef2dca31973dc3c08d6`
- `app/services/quality/evaluation.py`: `ad528baff0c45f6676170d4d3dfd6f68c4f804962daf8ab567eef4f085109aee`

## 独立评审与选择

主智能体逐题对冻结 calibration、SQLite 480 条问答/hash、最终排名及 gold 复算指标，通过；记录见 `parent-review.json`。唯一发生最终指标改善的是 V2E012：第三条 gold 在融合输出第 56 位，50 截断遗漏，100 保留后进入最终 Top-10。其余 47 道有 gold 的题无最终指标下降。平均重排增加 0.191 秒，约 31.223%。据此选择融合输出最多 100；每路召回仍 50，最终精排仍 10。

这是候选覆盖的局部证据，不能推断全部 test 或生成正确率一定提升。输入固定使用已审核的混合来源改写记录（56 条早期 Prompt、4 条最终 Prompt），只保证两个检索变体输入相同，不作为最新 Prompt 改写成功率。最终 300×4 会统一使用最终 Prompt，冻结 test 不用于再调参。
