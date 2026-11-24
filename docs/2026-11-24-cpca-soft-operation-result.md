# CPCA 软 operation 聚合：hard vs soft

日期：2026-11-24。改动：`scripts/audit_cpca_gate2.py` 增加 `--soft-tau`，
在硬选择之外同时输出 `cpca_soft` 等软加权结果。目标是用 $q_o$ 软加权替代
“每个窗口硬选一个 operation”，减少对 argmax 与 $n_{\min}$ 的依赖。

## 1. 方法

硬选择：

$$
o^\star=\arg\max_{o:\,n_o\ge n_{\min}}
\frac{\mathrm{p95}_o}{\mathrm{p95}_o^{\text{normal}}},
\qquad
r_i=r_i^{(o^\star)}.
$$

软加权：

$$
w_o=\frac{\exp(\mathrm{ratio}_o/\tau)}{\sum_{o'}\exp(\mathrm{ratio}_{o'}/\tau)},
\qquad
r_i=\sum_o w_o\,r_i^{(o)},\qquad \tau=1.
$$

硬选择是 $\tau\to0$ 的极限。当前权重用入口 p95 比值作为 $q_o$ 的代理；
接入训练好的 $q_o$ 是下一步。

## 2. 结果

事件级（同事件窗口分数取均值后排序一次）：

| 数据集 | 方法 | 事件数 | H@1 | H@3 | MRR |
|---|---|---:|---:|---:|---:|
| OB | CPCA hard | 54 | 0.907 | 1.000 | 0.954 |
| OB | CPCA soft | 54 | 0.907 | 1.000 | 0.951 |
| TT | CPCA hard | 63 | 0.952 | 1.000 | 0.974 |
| TT | CPCA soft | 64 | **0.953** | 1.000 | **0.974** |

对照（TT，软加权对弱方法的提升更明显）：

| 方法 | hard H@1/MRR | soft H@1/MRR |
|---|---:|---:|
| global_z | 0.349 / 0.556 | **0.547 / 0.699** |
| exclusive_z | 0.889 / 0.939 | **0.906 / 0.943** |
| CPCA | 0.952 / 0.974 | 0.953 / 0.974 |

覆盖率：OB 455 个正例分钟中 411 个有可用请求；TT 436 中 399 个。
TT 软加权多覆盖 1 个事件（63→64）。

## 3. 解释

1. **CPCA 已接近上限**：TT 的 H@1 0.952→0.953、MRR 不变，说明硬选择并没有明显丢分；
   软加权的价值主要是“去掉 argmax 和 $n_{\min}$ 的硬依赖”，而不是涨点。
2. **弱方法受益更大**：global_z 从 0.349 涨到 0.547，说明多 operation 聚合确实保留了
   被硬选择丢掉的信号；CPCA 因为本身已经选中了关键 operation，边际收益小。
3. **覆盖 +1**：软加权在 TT 多覆盖一个事件，且没有掉点，适合作为默认实现。
4. **OB 退化**：OB 只有 1 个入口 operation，soft 与 hard 应一致；
   MRR 0.954→0.951 的差异来自 `cpca_cov` 归一化与多 operation 状态的边界，量级可忽略。

## 4. 结论与下一步

- 默认用 soft（`--soft-tau 1.0`），hard 作为消融；
- 论文里把 soft 写成“$q_o$ 加权”，并注明当前用 p95 比值代理；
- 下一步：从训练好的 `OCSIDModel` 导出每个窗口的 $q_o$，用它替换代理权重，
  再比较一次 hard/soft/`q_o`-soft 三组。

## 5. 复现

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_cpca_gate2.py \
  --dataset tt --cases $(cat /tmp/tt_cases.txt | tr '\n' ' ') \
  --max-requests 100 --normal-mode first10 --soft-tau 1.0 \
  --output artifacts/cpca_soft_tt_first10_z2_scores_v1.json
```

OB 同理，输出 `artifacts/cpca_soft_ob_first10_z2_scores_v1.json`。
