# Gate 2 结果：CPCA 结构反事实归因可行性

日期：2026-11-23

状态：**有条件通过**。OB 上显著、机制清楚；TT 上略优但不显著（接近天花板）。
本结果支持继续做独立 OCSID，但不支持宣称“两个数据集都显著领先”。

## 1. 验证的问题

在树结构上做“恢复正常”的反事实重放，能否比现有 trace 单视图
（global duration、inclusive p95、exclusive/local）更好地把**注入服务**排到第一？

这是 OCSID 归因头的最小可行验证。不训练任何模型，只做离线审计。

## 2. 协议

- 数据集：RE2-OB（55 事件）、RE2-TT（64 事件），全部 90 个 case。
- 窗口：现有 `state + 1min` 正例分钟，事件级聚合（同事件窗口分数取均值后排序一次）。
- **operation 条件化**：每个窗口先选“当前 entry p95 / 正常 entry p95 比值最大
  且请求数 ≥ 20”的入口 operation，只在该 operation 的已完成请求上排序。
  OB 只有一个入口 operation，等价于全量。
- **正常参考（因果）**：每个 case 固定用前 10 分钟做 warm-up，不看注入时间。
  与“注入前全部分钟”的 oracle 版本对比，结果相同或更好。
- **基线**：按服务、按分钟计算 normal 中位数/MAD，测试窗口做 capped robust-z；
  分别是 global mean duration、inclusive p95、exclusive（local）mean。
- **未覆盖事件**：根因不在所选 operation 路径上时，按“排最后”计入，不丢弃。
  OB 有 1/55、TT 有 1/64 属于这种情况。

## 3. 主结果（事件级，未覆盖按最后）

### RE2-OB（55 事件）

| 方法 | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| **CPCA（population）** | **0.891** | **0.982** | **0.938** |
| CPCA（covered-only） | 0.982 | 0.982 | 0.983 |
| global_z | 0.164 | 0.436 | 0.397 |
| inclusive_z | 0.164 | 0.473 | 0.404 |
| exclusive_z（mean local） | 0.236 | 0.491 | 0.427 |
| exclusive_p95_z（Gate A 口径） | 0.236 | 0.491 | 0.442 |

### RE2-TT（64 事件）

| 方法 | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| **CPCA（population）** | **0.938** | **0.984** | **0.959** |
| CPCA（covered-only） | 0.938 | 0.984 | 0.959 |
| global_z | 0.344 | 0.672 | 0.547 |
| inclusive_z | 0.438 | 0.844 | 0.638 |
| exclusive_z（mean local） | 0.875 | 0.984 | 0.925 |
| exclusive_p95_z（Gate A 口径） | 0.656 | 0.922 | 0.789 |

参考：正式 LTFP 的事件级 H@1/MRR 是 OB `0.807/0.878`、TT `0.894/0.932`。
CPCA 在两边都更高，但**协议不同**（LTFP 是五折训练模型，CPCA 是零训练 + case 内正常参考），
不能直接当作“超过 LTFP”的正式结论。

## 4. 逐故障类型（H@1 / MRR）

### RE2-OB

| 故障 | n | CPCA | exclusive_z | global_z |
|---|---:|---:|---:|---:|
| CPU | 5 | **1.00 / 1.000** | 0.20 / 0.533 | 0.00 / 0.280 |
| delay | 13 | **0.69 / 0.846** | 0.23 / 0.402 | 0.23 / 0.462 |
| disk | 5 | **1.00 / 1.000** | 0.40 / 0.567 | 0.00 / 0.267 |
| loss | 15 | **0.87 / 0.906** | 0.20 / 0.330 | 0.33 / 0.480 |
| mem | 11 | **1.00 / 1.000** | 0.27 / 0.511 | 0.09 / 0.388 |
| socket | 6 | **1.00 / 1.000** | 0.17 / 0.363 | 0.00 / 0.274 |

### RE2-TT

| 故障 | n | CPCA | exclusive_z | global_z |
|---|---:|---:|---:|---:|
| CPU | 11 | **1.00 / 1.000** | 0.82 / 0.894 | 0.73 / 0.804 |
| delay | 11 | **1.00 / 1.000** | 0.91 / 0.955 | 0.09 / 0.430 |
| disk | 9 | 0.89 / 0.893 | 0.89 / 0.893 | 0.44 / 0.634 |
| loss | 11 | 0.82 / 0.894 | **0.91 / 0.939** | 0.09 / 0.326 |
| mem | 13 | **1.00 / 1.000** | 0.77 / 0.885 | 0.31 / 0.528 |
| socket | 9 | 0.89 / 0.944 | **1.00 / 1.000** | 0.44 / 0.590 |

## 5. 配对统计

对每个事件计算 reciprocal rank，按事件做 10,000 次 bootstrap：

| 数据集 | 对比 | ΔMRR | 95% CI | win/loss/tie |
|---|---|---:|---|---:|
| OB | CPCA − exclusive_z | **+0.521** | [+0.421, +0.614] | 40 / 1 / 13 |
| OB | CPCA − global_z | **+0.551** | [+0.449, +0.645] | 41 / 1 / 12 |
| OB | CPCA − exclusive_p95_z | **+0.505** | [+0.408, +0.596] | — |
| TT | CPCA − exclusive_z（最强基线） | +0.034 | [−0.013, +0.085] | 7 / 2 / 54 |
| TT | CPCA − exclusive_p95_z（Gate A） | **+0.173** | [+0.105, +0.246] | — |
| TT | CPCA − global_z | **+0.418** | [+0.333, +0.502] | 41 / 0 / 22 |

配对统计只在两侧都有排名的覆盖事件上计算（OB 54、TT 63），
因此与主表“未覆盖按最后”的口径略有差异。

结论：

- **OB：显著且量大。** 单视图基线几乎失效（H@1 0.16–0.24），CPCA 0.891。
- **TT：方向为正但落在噪声内。** exclusive_z 已经 0.875/0.925，CPCA 的增量
  只来自 7 个事件；不能据此声称 TT 上显著更好。

## 6. 机制解释（为什么 OB 上差这么多）

抽查 `currencyservice_delay/1`：delay 注入在 currencyservice，但等待时间出现在
checkout 的 local 里。exclusive 把 checkout 排第 1、根因排第 4；
CPCA 把这段“调用方等待”归给被调用方 currencyservice，根因排第 1。

这与 Gate A 的发现一致：exclusive 在 delay 上系统性排错，因为 post-gap 被算给了调用方。
CPCA 的修正不是拟合，而是归属规则变了。

TT 为什么增益小：TT 的 trace 更细（同一服务有 repository/DB 子 span），
delay 会同时抬高被注入服务自身的 local，所以 exclusive 本来就能排对；
CPCA 只在少数“纯网络等待”事件上补位。

## 7. 有效性检查

1. **因果参考 vs oracle**：在覆盖事件口径上，TT 用 first10 的 CPCA H@1 为 0.952，
   pre 版本为 0.922；OB 两者完全一致。说明“用注入前 10 分钟”不是结果的主要来源。
2. **覆盖率**：正例分钟中有请求被打分的比例 OB 90.3%、TT 91.5%；
   没有整个 case 丢失。未覆盖的 2 个事件按最后计入。
3. **operation 选择**：每窗口只选一个入口 operation，避免把无关 operation 混在一起。
   OB 只有一个入口 operation，因此 OB 的增益不来自 operation 选择，而来自归属规则。
4. **参考覆盖**：正常参考按 `(service, operation)` 和边分别估计；
   缺失时回退到观测值（该服务反事实为 0），不会凭空制造分数。

## 8. 局限（必须写进论文的 threats to validity）

1. **不是五折训练对比**：CPCA 零训练、确定性；LTFP 是五折训练。
   两者事件集合相同，但信息与协议不同，不能直接下“超过 LTFP”的结论。
2. **基线已对齐 Gate A**：增加了 `exclusive_p95_z`（operation 内 local p95 + capped robust-z）。
   结果对基线口径**不敏感**：OB 上 CPCA 相对 p95 口径 ΔMRR `+0.505`；
   TT 上相对 p95 口径 `+0.173` 显著，但相对均值 local 口径 `+0.034` 不显著。
   论文必须同时报告这两个 exclusive 变体，不能只挑一个。
3. **单次确定性运行**：没有种子方差；CI 只反映事件抽样，不反映训练方差。
4. **事件级聚合是回顾性的**：同事件窗口分数平均后再排序，不是在线逐分钟决策。
5. **CPCA（covered-only）在 OB 上 0.982/0.983 偏高**：低覆盖服务的分子分母都很小，
   容易饱和。主结果应报 population 版本（0.891/0.938），covered-only 只作诊断。
6. **loss/socket 在 TT 上不是最优**：loss CPCA 0.82 vs exclusive 0.91，
   socket 0.89 vs 1.00。说明“结构反事实”不是万能的，需按故障类型报告。
7. **未做 top-1/top-2 margin 分析**：1.000 这种数字需要看分差，不能只看排名。

## 9. 判定

**Gate 2 有条件通过。**

- 支持继续：OB 上机制清楚、效应大、配对 CI 不含 0；
  TT 上不劣于最强单视图，且平均略优。
- 不支持：宣称“两数据集都显著领先”，或直接用 CPCA 替代 LTFP 的预测头。
- 下一步必须先补两件事，再写 OCSID V1：
  1. 完全对齐 Gate A 的 p95/robust-z 基线，确认 OB 优势仍成立；
  2. 对 OB 的 covered-only 做 margin 分析，排除小分母饱和。

## 10. 对 OCSID V1 的影响

1. **归因头确定用 CPCA**，不用学习式 restoration；学习式 restoration 没必要再试。
2. **预测头仍需要**：CPCA 只解决“给定正例窗口，谁负责”，
   不解决“下一分钟是否故障”。V1 的 operation 风险头 + source/impact 编码器保留。
3. **融合权重 λ 应该小**：TT 上 CPCA 与 exclusive 高度一致（54/63 tie），
   说明结构证据和局部证据冗余；融合只在 OB 类“纯网络等待”场景有增量。
4. **论文故事要按数据集分开写**：
   - OB：单入口 operation、网络等待主导 → CPCA 显著更好；
   - TT：多入口 operation、局部信号强 → CPCA 不劣，增益集中在 CPU/mem/delay 少数事件；
   - loss/socket：结构证据不足，需显式报告为边界。

## 11. 复现

```bash
# OB
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_cpca_gate2.py \
  --dataset ob --cases $(cat /tmp/ob_cases.txt | tr '\n' ' ') \
  --max-requests 100 --normal-mode first10 \
  --output artifacts/cpca_gate2_ob_first10_z_v1.json

# TT
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_cpca_gate2.py \
  --dataset tt --cases $(cat /tmp/tt_cases.txt | tr '\n' ' ') \
  --max-requests 100 --normal-mode first10 \
  --output artifacts/cpca_gate2_tt_first10_z_v1.json
```

case 列表来自：

```bash
.venv/bin/python - <<'EOF' > /tmp/tt_cases.txt
import zipfile
z = zipfile.ZipFile('data/RCAEval/RE2-TT.zip')
for n in z.namelist():
    if n.endswith('/traces.csv'):
        print(n[len('RE2-TT/'):-len('/traces.csv')])
EOF
```
