# ch04 四策略评估

模式：live

同模型裁判基线，存在自评偏差
语料 chunk：480；生成：deepseek-chat；裁判：deepseek-chat

| 策略 | 题数 | Recall@1 | Recall@5 | Recall@10 | 完整证据@10 | MRR | 答案正确率 | Faithfulness | 错误 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 300 | 0.756 | 0.941 | 0.953 | 0.917 | 0.932 | 0.923 | 1.000 | 0 |
| bm25 | 300 | 0.640 | 0.783 | 0.783 | 0.750 | 0.789 | 0.790 | 1.000 | 0 |
| hybrid | 300 | 0.752 | 0.948 | 0.950 | 0.917 | 0.936 | 0.927 | 1.000 | 0 |
| hybrid_rerank | 300 | 0.776 | 0.985 | 0.996 | 0.988 | 0.936 | 0.980 | 1.000 | 0 |

## dense 按类型

| 桶 | Recall@5 | 完整证据@10 | MRR | 正确率 | Faithfulness | 未知拒答率 | 已知误拒答率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_policy | 1.000 | 1.000 | 1.000 | 0.983 | 1.000 | N/A | 0.017 |
| B_model | 1.000 | 1.000 | 0.799 | 0.983 | 1.000 | N/A | 0.017 |
| C_colloquial | 1.000 | 1.000 | 0.975 | 1.000 | 1.000 | N/A | 0.000 |
| D_unknown | N/A | N/A | N/A | 1.000 | N/A | 1.000 | N/A |
| E_multi | 0.764 | 0.667 | 0.953 | 0.650 | 1.000 | N/A | 0.350 |

## bm25 按类型

| 桶 | Recall@5 | 完整证据@10 | MRR | 正确率 | Faithfulness | 未知拒答率 | 已知误拒答率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_policy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | N/A | 0.000 |
| B_model | 1.000 | 1.000 | 0.933 | 0.983 | 1.000 | N/A | 0.017 |
| C_colloquial | 0.333 | 0.333 | 0.333 | 0.333 | 1.000 | N/A | 0.667 |
| D_unknown | N/A | N/A | N/A | 1.000 | N/A | 1.000 | N/A |
| E_multi | 0.800 | 0.667 | 0.890 | 0.633 | 1.000 | N/A | 0.367 |

## hybrid 按类型

| 桶 | Recall@5 | 完整证据@10 | MRR | 正确率 | Faithfulness | 未知拒答率 | 已知误拒答率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_policy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | N/A | 0.000 |
| B_model | 1.000 | 1.000 | 0.908 | 1.000 | 1.000 | N/A | 0.000 |
| C_colloquial | 1.000 | 1.000 | 0.867 | 0.967 | 1.000 | N/A | 0.000 |
| D_unknown | N/A | N/A | N/A | 1.000 | N/A | 1.000 | N/A |
| E_multi | 0.792 | 0.667 | 0.971 | 0.667 | 1.000 | N/A | 0.333 |

## hybrid_rerank 按类型

| 桶 | Recall@5 | 完整证据@10 | MRR | 正确率 | Faithfulness | 未知拒答率 | 已知误拒答率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_policy | 1.000 | 1.000 | 1.000 | 0.983 | 1.000 | N/A | 0.017 |
| B_model | 1.000 | 1.000 | 1.000 | 0.983 | 1.000 | N/A | 0.017 |
| C_colloquial | 1.000 | 1.000 | 0.751 | 0.983 | 1.000 | N/A | 0.017 |
| D_unknown | N/A | N/A | N/A | 1.000 | N/A | 1.000 | N/A |
| E_multi | 0.939 | 0.950 | 0.992 | 0.950 | 1.000 | N/A | 0.050 |

## Test 同题配对差异

| 左 vs 右 | 指标 | 配对数 | 左胜 | 平 | 右胜 | 右减左均值 |
|---|---|---:|---:|---:|---:|---:|
| dense_vs_bm25 | recall@1 | 192 | 41 | 133 | 18 | -0.113 |
| dense_vs_bm25 | recall@5 | 192 | 32 | 155 | 5 | -0.157 |
| dense_vs_bm25 | recall@10 | 192 | 36 | 154 | 2 | -0.170 |
| dense_vs_bm25 | all_evidence@10 | 192 | 32 | 160 | 0 | -0.167 |
| dense_vs_bm25 | mrr | 192 | 43 | 129 | 20 | -0.139 |
| dense_vs_bm25 | correctness | 240 | 35 | 203 | 2 | -0.138 |
| dense_vs_hybrid | recall@1 | 192 | 11 | 169 | 12 | -0.002 |
| dense_vs_hybrid | recall@5 | 192 | 0 | 188 | 4 | 0.007 |
| dense_vs_hybrid | recall@10 | 192 | 4 | 186 | 2 | -0.003 |
| dense_vs_hybrid | all_evidence@10 | 192 | 0 | 192 | 0 | 0.000 |
| dense_vs_hybrid | mrr | 192 | 11 | 162 | 19 | 0.007 |
| dense_vs_hybrid | correctness | 240 | 2 | 236 | 2 | 0.000 |
| dense_vs_hybrid_rerank | recall@1 | 192 | 14 | 158 | 20 | 0.021 |
| dense_vs_hybrid_rerank | recall@5 | 192 | 0 | 175 | 17 | 0.043 |
| dense_vs_hybrid_rerank | recall@10 | 192 | 0 | 176 | 16 | 0.042 |
| dense_vs_hybrid_rerank | all_evidence@10 | 192 | 0 | 179 | 13 | 0.068 |
| dense_vs_hybrid_rerank | mrr | 192 | 16 | 156 | 20 | 0.006 |
| dense_vs_hybrid_rerank | correctness | 240 | 2 | 223 | 15 | 0.054 |
| bm25_vs_hybrid | recall@1 | 192 | 6 | 156 | 30 | 0.111 |
| bm25_vs_hybrid | recall@5 | 192 | 1 | 159 | 32 | 0.164 |
| bm25_vs_hybrid | recall@10 | 192 | 0 | 160 | 32 | 0.167 |
| bm25_vs_hybrid | all_evidence@10 | 192 | 0 | 160 | 32 | 0.167 |
| bm25_vs_hybrid | mrr | 192 | 7 | 142 | 43 | 0.146 |
| bm25_vs_hybrid | correctness | 240 | 0 | 207 | 33 | 0.138 |
| bm25_vs_hybrid_rerank | recall@1 | 192 | 1 | 161 | 30 | 0.134 |
| bm25_vs_hybrid_rerank | recall@5 | 192 | 0 | 146 | 46 | 0.200 |
| bm25_vs_hybrid_rerank | recall@10 | 192 | 0 | 145 | 47 | 0.212 |
| bm25_vs_hybrid_rerank | all_evidence@10 | 192 | 0 | 147 | 45 | 0.234 |
| bm25_vs_hybrid_rerank | mrr | 192 | 1 | 145 | 46 | 0.145 |
| bm25_vs_hybrid_rerank | correctness | 240 | 2 | 190 | 48 | 0.192 |
| hybrid_vs_hybrid_rerank | recall@1 | 192 | 4 | 179 | 9 | 0.023 |
| hybrid_vs_hybrid_rerank | recall@5 | 192 | 0 | 177 | 15 | 0.036 |
| hybrid_vs_hybrid_rerank | recall@10 | 192 | 0 | 177 | 15 | 0.045 |
| hybrid_vs_hybrid_rerank | all_evidence@10 | 192 | 0 | 179 | 13 | 0.068 |
| hybrid_vs_hybrid_rerank | mrr | 192 | 16 | 167 | 9 | -0.002 |
| hybrid_vs_hybrid_rerank | correctness | 240 | 2 | 223 | 15 | 0.054 |

拒答无事实声明记 N/A，不计为忠实度通过。正例误拒答的正确性为0；未知题按显式拒答判定。正确性与证据忠实度分开，不共享 ground-truth 输入。
错误按出错行及各裁判分别计数；两项裁判独立尝试，有效判定不受另一裁判失败影响。分母及事实覆盖率保存在JSON。检索指标保留已完成检索的行。
类型、难度、挑战标签、test及calibration分组分别保存于JSON。受控虚构语料尚未经业务专家独立复核；不得按test成绩筛题或调参。
