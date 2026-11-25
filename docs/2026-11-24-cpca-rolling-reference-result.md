# CPCA 滚动正常参考：固定 vs 滚动

日期：2026-11-24。目标：把固定的“case 前 10 分钟”正常参考换成
**$t$ 之前最近 $W$ 个健康分钟**的因果滚动参考，并向固定先验收缩，
看是否能提升根因排序、回应“前 10 分钟未必正常”的质疑。

## 1. 做法

对每个正例分钟 $t$，取最近 $W$ 个 `failure=0` 的分钟（可加 guard band，
排除 $t$ 前 1–2 分钟），对每个 `(service, operation)` 的 local 和每条边的 pre/post
计算中位数，再向固定先验收缩：

$$
\hat{x}_t=\frac{n}{n+k}\,\mathrm{med}_{\text{rolling}}
+\frac{k}{n+k}\,\mathrm{med}_{\text{fixed}},
$$

其中 $n$ 是滚动窗口内的样本数，$k$ 是收缩强度。
$n=0$ 时退化为固定先验。

实现：`scripts/audit_cpca_gate2.py` 新增
`--reference rolling --reference-window W --shrink-k K --reference-guard G`。
默认仍是 `--reference fixed`。

## 2. 结果

事件级：

| 数据集 | 参考 | 方法 | 事件数 | H@1 | H@3 | MRR |
|---|---|---|---:|---:|---:|---:|
| OB | fixed first10 | CPCA soft | 54 | 0.907 | 1.000 | 0.951 |
| OB | rolling (W=10,k=5) | CPCA soft | 54 | 0.907 | 1.000 | 0.948 |
| TT | fixed first10 | CPCA hard | 63 | 0.952 | 1.000 | 0.974 |
| TT | fixed first10 | CPCA soft | 64 | **0.953** | **1.000** | **0.974** |
| TT | rolling (W=10,k=5) | CPCA soft | 64 | 0.938 | 0.984 | 0.961 |
| TT | rolling + guard2 (W=8,k=10) | CPCA soft | 64 | 0.938 | 0.984 | 0.961 |
| TT | rolling + guard2 | CPCA hard | 63 | 0.937 | 0.984 | 0.962 |

## 3. 解释

1. **滚动参考没有更好，反而更差**：TT soft 从 0.953/0.974 降到 0.938/0.961。
2. **guard band 有改善但不够**：排除 $t$ 前 2 分钟后，硬选择从 0.921 回到 0.937，
   说明事件前的“健康分钟”确实已被早期退化污染；但即使这样仍低于固定参考。
3. **原因是标签与故障的时间错位**：`failure=0` 只表示“该分钟未达到用户可感知阈值”，
   不代表根因服务已经正常。事件前 1–10 分钟往往已经有局部异常，
   滚动参考会把这些异常吸收进基线，导致 $\Delta_i$ 被低估。
4. **固定前 10 分钟更稳**：它是注入前的干净窗口，虽然可能不代表当前负载，
   但至少不会被故障前兆污染。对“恢复后能移除多少”这个估计量，
   干净比“贴近当前”更重要。

## 4. 结论

- **默认保留 `--reference fixed`（case 前 10 分钟）**；
- 滚动参考作为**敏感性分析**写进论文，说明我们考虑过非平稳性，
  但数据表明固定预热参考更稳；
- 论文表述建议：
  > “We also tested a causal rolling reference with shrinkage; it did not improve
  > over the fixed pre-injection reference (TT 0.938 vs 0.953), because healthy
  > minutes before the event already contain early degradation. We therefore keep
  > the fixed warm-up reference and report the rolling variant as a sensitivity check.”

## 5. 复现

```bash
# 固定参考（默认）
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_cpca_gate2.py \
  --dataset tt --cases $(cat /tmp/tt_cases.txt | tr '\n' ' ') \
  --max-requests 100 --normal-mode first10 --reference fixed \
  --output artifacts/cpca_soft_tt_first10_z2_scores_v1.json

# 滚动参考
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_cpca_gate2.py \
  --dataset tt --cases $(cat /tmp/tt_cases.txt | tr '\n' ' ') \
  --max-requests 100 --normal-mode first10 --reference rolling \
  --reference-window 10 --shrink-k 5 \
  --output artifacts/cpca_rolling_tt_first10_z2_scores_v1.json
```
