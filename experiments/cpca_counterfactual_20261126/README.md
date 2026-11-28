# CPCA 反事实实验与文档（2026-11-26）

当前位置是本轮实验的统一维护入口，主模型仍在 `src/ocsid/`。从 MyDocs 迁入时保留原文件作为备份；本目录的代码已改为仓库相对路径，不依赖 MyDocs。迁移只纳入正式汇总，不纳入中断跑法、smoke 和冗余单案例输出。

## 先看什么

- `COUNTERFACTUAL_V2_REVIEW.md`：最新方法公式、同设置五组对照、创新性判断。
- `PAPER_ACCEPTANCE_REVIEW_20261126.md`：面向投稿的缺口分级、最小收尾实验、告警门控新结果和定型标准。
- `OCSID_CPCA_Method.md`：历史方法章节草稿，含修正提醒。
- `DISTRIBUTIONAL_CPCA_RESULTS.md`：早期实验记录。旧 TT 比较混入汇总设置差异，以最新报告为准。
- `FINAL_RESULTS.md`、`FINAL_RESULTS.json`：此前主实验的背景汇总快照。
- `SHAPLEYIQ_SAGE_*`：此前近邻对比的背景记录。本目录未迁入这些背景记录所依赖的全套上游 runner。

## 代码与数据

- `experiment_paths.py`：仓库根目录、实验根目录和输入目录。
- `refine_cpca.py`：完整请求匹配、逐服务重放及五组同请求消融。
- `run_distributional_cpca.py`：早期 span 匹配 runner，同时提供 v2 复用函数；迁入后其 baseline 指标改用同设置 `cpca_uniform`。保存的 v1 JSON 仍保留原始历史指标，不以其中旧比较作为结论。
- `summarize_counterfactual_v2.py`：一致性核验与报告生成。
- `evaluate_alert_conditioned_rca.py`：按每折验证阈值连接 OCSID 告警与同分钟 CPCA，分别报告条件 RCA 与漏报告警计零的端到端指标。
- `inputs/`：固定真实正例评价集合所在的两个预测文件。
- `attribution/`：原 median 对照审计。
- `distributional_cpca/`：上一版正式结果。
- `counterfactual_v2/`：最新正式结果、示例与核验输出。
- `MIGRATION_MANIFEST.json`：迁入文件的来源与 SHA-256。

大体积输入、结果 JSON 由本目录 `.gitignore` 排除；本地文件已完整迁入，Git 中保留代码、文档和迁移清单。训练模型、原始数据仍使用仓库原有目录。

## 在仓库根目录使用

```bash
.venv/bin/python -B experiments/cpca_counterfactual_20261126/refine_cpca.py --help
.venv/bin/python -B experiments/cpca_counterfactual_20261126/summarize_counterfactual_v2.py
```

已有正式结果受到防覆盖保护。重新运行时指定独立输出路径：

```bash
.venv/bin/python -B experiments/cpca_counterfactual_20261126/refine_cpca.py --dataset ob --output experiments/cpca_counterfactual_20261126/reruns/ob/results.json
.venv/bin/python -B experiments/cpca_counterfactual_20261126/refine_cpca.py --dataset tt --output experiments/cpca_counterfactual_20261126/reruns/tt/results.json
```

上述命令读取原始 trace，执行完整审计。报告生成器默认核对已迁入的正式结果，不会自动选取 reruns 结果。单案例审计文件是中间输出；以最终 results.json 中标注的五组分数为准。

## 研究结论边界

RCA 评价覆盖根因标签有效的所有真实正例，不是按预测成功告警筛选。请求级重放没有额外排序收益；经验范围不是因果置信区间。当前建议维持原 median + uniform 主排序，均值参考作消融，完整请求重放作解释；最终版本尚待用户决定。
