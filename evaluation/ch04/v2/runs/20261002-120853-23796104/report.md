# ch04 四策略评估

模式：live

同模型裁判基线，存在自评偏差
语料 chunk：480；生成：deepseek-chat；裁判：deepseek-chat

| 策略 | 题数 | Recall@1 | Recall@5 | Recall@10 | 完整证据@10 | MRR | 答案正确率 | Faithfulness | 错误 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 300 | 0.756 | 0.941 | 0.951 | 0.917 | 0.933 | 0.933 | 0.997 | 0 |
| bm25 | 300 | 0.640 | 0.783 | 0.783 | 0.750 | 0.789 | 0.787 | 0.999 | 0 |
| hybrid | 300 | 0.750 | 0.948 | 0.950 | 0.917 | 0.935 | 0.930 | 1.000 | 0 |
| hybrid_rerank | 300 | 0.718 | 0.982 | 0.990 | 0.971 | 0.907 | 0.963 | 1.000 | 0 |

## dense 按类型

| 桶 | Recall@5 | 完整证据@10 | MRR | 正确率 | Faithfulness | 未知拒答率 | 已知误拒答率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_policy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | N/A | 0.000 |
| B_model | 1.000 | 1.000 | 0.805 | 1.000 | 0.993 | N/A | 0.000 |
| C_colloquial | 1.000 | 1.000 | 0.975 | 1.000 | 0.994 | N/A | 0.000 |
| D_unknown | N/A | N/A | N/A | 1.000 | N/A | 1.000 | N/A |
| E_multi | 0.764 | 0.667 | 0.953 | 0.667 | 1.000 | N/A | 0.333 |

## bm25 按类型

| 桶 | Recall@5 | 完整证据@10 | MRR | 正确率 | Faithfulness | 未知拒答率 | 已知误拒答率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_policy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | N/A | 0.000 |
| B_model | 1.000 | 1.000 | 0.933 | 0.967 | 1.000 | N/A | 0.033 |
| C_colloquial | 0.333 | 0.333 | 0.333 | 0.333 | 0.993 | N/A | 0.667 |
| D_unknown | N/A | N/A | N/A | 1.000 | N/A | 1.000 | N/A |
| E_multi | 0.800 | 0.667 | 0.890 | 0.633 | 1.000 | N/A | 0.367 |

## hybrid 按类型

| 桶 | Recall@5 | 完整证据@10 | MRR | 正确率 | Faithfulness | 未知拒答率 | 已知误拒答率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_policy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | N/A | 0.000 |
| B_model | 1.000 | 1.000 | 0.911 | 1.000 | 1.000 | N/A | 0.000 |
| C_colloquial | 1.000 | 1.000 | 0.867 | 0.983 | 1.000 | N/A | 0.000 |
| D_unknown | N/A | N/A | N/A | 1.000 | N/A | 1.000 | N/A |
| E_multi | 0.792 | 0.667 | 0.963 | 0.667 | 1.000 | N/A | 0.333 |

## hybrid_rerank 按类型

| 桶 | Recall@5 | 完整证据@10 | MRR | 正确率 | Faithfulness | 未知拒答率 | 已知误拒答率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_policy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | N/A | 0.000 |
| B_model | 1.000 | 1.000 | 0.883 | 0.967 | 1.000 | N/A | 0.033 |
| C_colloquial | 1.000 | 1.000 | 0.751 | 0.967 | 1.000 | N/A | 0.033 |
| D_unknown | N/A | N/A | N/A | 1.000 | N/A | 1.000 | N/A |
| E_multi | 0.928 | 0.883 | 0.992 | 0.883 | 1.000 | N/A | 0.117 |

## Test 同题配对差异

| 左 vs 右 | 指标 | 配对数 | 左胜 | 平 | 右胜 | 右减左均值 |
|---|---|---:|---:|---:|---:|---:|
| dense_vs_bm25 | recall@1 | 192 | 41 | 133 | 18 | -0.113 |
| dense_vs_bm25 | recall@5 | 192 | 32 | 155 | 5 | -0.157 |
| dense_vs_bm25 | recall@10 | 192 | 35 | 155 | 2 | -0.168 |
| dense_vs_bm25 | all_evidence@10 | 192 | 32 | 160 | 0 | -0.167 |
| dense_vs_bm25 | mrr | 192 | 43 | 130 | 19 | -0.140 |
| dense_vs_bm25 | correctness | 240 | 35 | 205 | 0 | -0.146 |
| dense_vs_hybrid | recall@1 | 192 | 12 | 168 | 12 | -0.004 |
| dense_vs_hybrid | recall@5 | 192 | 0 | 188 | 4 | 0.007 |
| dense_vs_hybrid | recall@10 | 192 | 3 | 187 | 2 | -0.002 |
| dense_vs_hybrid | all_evidence@10 | 192 | 0 | 192 | 0 | 0.000 |
| dense_vs_hybrid | mrr | 192 | 12 | 162 | 18 | 0.004 |
| dense_vs_hybrid | correctness | 240 | 1 | 239 | 0 | -0.004 |
| dense_vs_hybrid_rerank | recall@1 | 192 | 14 | 168 | 10 | -0.031 |
| dense_vs_hybrid_rerank | recall@5 | 192 | 0 | 175 | 17 | 0.041 |
| dense_vs_hybrid_rerank | recall@10 | 192 | 0 | 176 | 16 | 0.038 |
| dense_vs_hybrid_rerank | all_evidence@10 | 192 | 0 | 182 | 10 | 0.052 |
| dense_vs_hybrid_rerank | mrr | 192 | 16 | 162 | 14 | -0.021 |
| dense_vs_hybrid_rerank | correctness | 240 | 4 | 226 | 10 | 0.025 |
| bm25_vs_hybrid | recall@1 | 192 | 6 | 157 | 29 | 0.109 |
| bm25_vs_hybrid | recall@5 | 192 | 1 | 159 | 32 | 0.164 |
| bm25_vs_hybrid | recall@10 | 192 | 0 | 160 | 32 | 0.167 |
| bm25_vs_hybrid | all_evidence@10 | 192 | 0 | 160 | 32 | 0.167 |
| bm25_vs_hybrid | mrr | 192 | 7 | 142 | 43 | 0.144 |
| bm25_vs_hybrid | correctness | 240 | 0 | 206 | 34 | 0.142 |
| bm25_vs_hybrid_rerank | recall@1 | 192 | 9 | 155 | 28 | 0.082 |
| bm25_vs_hybrid_rerank | recall@5 | 192 | 0 | 145 | 47 | 0.198 |
| bm25_vs_hybrid_rerank | recall@10 | 192 | 0 | 145 | 47 | 0.207 |
| bm25_vs_hybrid_rerank | all_evidence@10 | 192 | 0 | 150 | 42 | 0.219 |
| bm25_vs_hybrid_rerank | mrr | 192 | 9 | 139 | 44 | 0.118 |
| bm25_vs_hybrid_rerank | correctness | 240 | 2 | 195 | 43 | 0.171 |
| hybrid_vs_hybrid_rerank | recall@1 | 192 | 8 | 180 | 4 | -0.027 |
| hybrid_vs_hybrid_rerank | recall@5 | 192 | 0 | 176 | 16 | 0.034 |
| hybrid_vs_hybrid_rerank | recall@10 | 192 | 0 | 177 | 15 | 0.040 |
| hybrid_vs_hybrid_rerank | all_evidence@10 | 192 | 0 | 182 | 10 | 0.052 |
| hybrid_vs_hybrid_rerank | mrr | 192 | 20 | 167 | 5 | -0.025 |
| hybrid_vs_hybrid_rerank | correctness | 240 | 4 | 225 | 11 | 0.029 |

拒答无事实声明记 N/A，不计为忠实度通过。正例误拒答的正确性为0；未知题按显式拒答判定。正确性与证据忠实度分开，不共享 ground-truth 输入。
错误按出错行及各裁判分别计数；两项裁判独立尝试，有效判定不受另一裁判失败影响。分母及事实覆盖率保存在JSON。检索指标保留已完成检索的行。
类型、难度、挑战标签、test及calibration分组分别保存于JSON。受控虚构语料尚未经业务专家独立复核；不得按test成绩筛题或调参。
