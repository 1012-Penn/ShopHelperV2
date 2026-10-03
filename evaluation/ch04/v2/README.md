# ch04 评估集 v2

300题，五桶各60；每桶easy/medium/hard各20。60 calibration / 240 test，按全局20个场景家族分开，同系列型号和政策跨桶不跨split。

480个有效chunk，80个虚构商品型号及160条限定场景政策。全部MX型号、配件编号和活动为受控虚构资料，只进隔离评估库；不得作为商家真实承诺。题目含近似后缀、口语、地域/渠道/状态例外、拒答与2至3条必要证据；140题带实际品类过滤。

标注有必要事实、gold、干扰来源、应否拒答、拒答依据、家族和挑战标签。manifest记录冻结hash。标注作者和核验者是同一助手，未做业务专家盲审；通用规则模板跨家族仍有相似性。

本版在任何真实测试分数出现前冻结，不按成绩筛题。只修正可证明的标注错误，必须另记修订和新hash；不得拿test结果选择阈值。首轮固定生产阈值0.05。

命令：

```bash
python3 -m scripts.validate_ch04_dataset --dataset-dir evaluation/ch04/v2
python3 -m scripts.evaluate_ch04 --live --dataset-dir evaluation/ch04/v2 --split all --workers 8
```

默认隔离SQLite为本目录eval.db，Milvus为knowledge_ch04_eval_v2；生产和v1不会被覆盖。环境中显式EVAL配置可覆盖默认，报告会记录。
