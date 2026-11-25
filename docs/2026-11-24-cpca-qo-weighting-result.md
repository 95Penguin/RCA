# CPCA 用训练好的 q_o 加权：负结果

日期：2026-11-24。目标：把 `OCSIDModel` 的 operation 风险 $q_o$ 作为 CPCA 的
operation 权重，替代 p95 比值代理，看是否能提升根因排序。

## 1. 做法

1. `scripts/train_ocsid_joint.py` 现在把每个测试窗口的 `operation_scores`（$q_o$）
   和 operation 词表一起写入结果 JSON；
2. 训练 seed 42 五折，得到 `artifacts/ocsid_joint_{ob,tt}_seed42_qo_v3.json`；
3. `scripts/audit_cpca_gate2.py` 新增 `--qo-table`，按窗口读取 $q_o$，
   对“有 $q_o$ 定义的 operation”做 softmax 加权：

$$
w_o=\frac{\exp(q_o/\tau)}{\sum_{o'}\exp(q_{o'}/\tau)},\qquad
r_i=\sum_o w_o\,r_i^{(o)}.
$$

没有 $q_o$ 的候选 operation 不参与加权；若一个都没有，回退到 p95 soft。

## 2. 结果

事件级：

| 数据集 | 方法 | 事件数 | H@1 | H@3 | MRR |
|---|---|---:|---:|---:|---:|
| OB | CPCA hard | 54 | 0.907 | 1.000 | 0.954 |
| OB | CPCA p95-soft | 54 | 0.907 | 1.000 | 0.951 |
| OB | CPCA q_o-soft | 54 | 0.907 | 1.000 | 0.954 |
| TT | CPCA hard | 63 | 0.952 | 1.000 | 0.974 |
| TT | CPCA p95-soft | 64 | 0.953 | 1.000 | 0.974 |
| TT | CPCA q_o-soft | 64 | 0.953 | 1.000 | 0.974 |

覆盖：OB 54 个事件；TT 64 个事件（soft/q 比 hard 多 1 个）。

## 3. 解释

1. **TT 上 $q_o$ 与 p95-soft 完全相同**：两者都选到了相同的 operation 组合，
   $q_o$ 没有带来额外区分度。
2. **OB 上 $q_o$ 退化为 hard**：OB 只有 1 个入口 operation，$q_o$ 是单个标量，
   无法在多个 operation 之间加权；过滤后只剩一个候选，等价于 hard。
3. **原因不意外**：$q_o$ 的训练目标是“下一分钟 operation 是否退化”，
   不是“这个 operation 是否是根因路径”。它与 CPCA 的责任分目标不同，
   所以不能指望它当归因权重。
4. **这反过来支持两阶段解耦**：学习式 operation 风险适合做预测，
   结构责任分适合做归因；把两者硬接不会更好。

## 4. 结论

- **默认仍用 p95-soft**（`--soft-tau 1.0`），$q_o$ 不作为归因权重；
- $q_o$ 保留在预测侧，作为 operation 风险头与可解释输出；
- 论文里可以写一句：“用训练好的 $q_o$ 作为归因权重不优于 p95 比值
  （TT 0.953/0.974，与 soft 相同），因此本文不让学习式风险进入 CPCA 的权重。”
  这能提前回答“为什么不端到端”的质疑。

## 5. 复现

```bash
# 1) 训练并导出 q_o（seed 42，五折）
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train_ocsid_joint.py \
  --dataset tt --folds 0 1 2 3 4 --epochs 30 --seed 42 \
  --output artifacts/ocsid_joint_tt_seed42_qo_v3.json

# 2) 带 q_o 的 CPCA 审计
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_cpca_gate2.py \
  --dataset tt --cases $(cat /tmp/tt_cases.txt | tr '\n' ' ') \
  --max-requests 100 --normal-mode first10 --soft-tau 1.0 \
  --qo-table artifacts/ocsid_joint_tt_seed42_qo_v3.json \
  --output artifacts/cpca_qo_tt_first10_z2_scores_v2.json
```

OB 同理，输出 `artifacts/cpca_qo_ob_first10_z2_scores_v2.json`。
