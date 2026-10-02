# ch04 标注评估集 v1

300 条，五桶各 60：政策、具体型号、口语、未知、跨片；每桶 easy/medium/hard 各 20。每桶首四个 topic（12 题）为 calibration，其余 48 为 test，总计 60/240。topic 在桶内不跨 split；跨桶涉及相同政策的情况需以全量报告注明，不将此集合声称为无语义泄漏的外部盲测。

标注来源为稳定 source_key，配必要事实和应否拒答。政策来自 ch03 受控业务基线；XH-300 至 XH-319 均为虚构测试商品，只入隔离评估库。数据由助手构建并核对原文，尚未经人工业务专家复核。difficulty 按直问、条件场景、否定/多条件划分。此集覆盖明确业务类型但不是生产用户分布。

运行 `python3 -m scripts.validate_ch04_dataset` 验证计数、标注、来源与分桶。运行 `python3 -m scripts.evaluate_ch04 --live --split all` 对全部 300 题比较四策略；fixture 数字不代表真实效果。
