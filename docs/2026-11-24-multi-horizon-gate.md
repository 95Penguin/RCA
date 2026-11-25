# 多 horizon Gate 0：3/5 分钟还有根因信号吗？

日期：2026-11-24。目的：在投入 hazard 头 / trailing CPCA 之前，先确认
在决策时刻 $t$，根因服务是否已经出现可观测异常。若没有，3/5 分钟的 CPCA
本质上不可做，不是模型问题。

## 1. 口径

对每个 horizon $h\in\{1,3,5\}$ 与任务：

- `state@h`：预测 $t+h$ 的精确状态；
- `first-visible@h`：预测 $(t,t+h]$ 内首次出现故障。

对每个正例窗口，取根因服务 `root`，用该 case 前 10 分钟的均值/标准差做基线，
若根因服务的面板特征满足以下任一条件，则记为“有信号”：

- 任一特征 $|z|>3$；
- `entry_p95_ms` 超过基线 1.5 倍。

统计窗口级与事件级的信号比例。

## 2. 结果

### RE2-TT

| h | 任务 | 正例窗口 | 事件 | 窗口有信号 | 事件有信号 |
|---:|---|---:|---:|---:|---:|
| 1 | state | 436 | 64 | 0.991 | 1.000 |
| 3 | state | 383 | 62 | 0.909 | 0.935 |
| 5 | state | 385 | 63 | 0.704 | 0.857 |
| 1 | first-visible | 53 | 53 | 0.925 | 0.925 |
| 3 | first-visible | 72 | 53 | 0.944 | 0.925 |
| 5 | first-visible | 80 | 53 | 0.950 | 0.925 |

### RE2-OB

| h | 任务 | 正例窗口 | 事件 | 窗口有信号 | 事件有信号 |
|---:|---|---:|---:|---:|---:|
| 1 | state | 455 | 55 | 0.967 | 1.000 |
| 3 | state | 406 | 55 | 0.865 | 0.964 |
| 5 | state | 413 | 55 | 0.644 | 0.855 |
| 1 | first-visible | 51 | 51 | 0.765 | 0.765 |
| 3 | first-visible | 62 | 51 | 0.806 | 0.765 |
| 5 | first-visible | 70 | 51 | 0.800 | 0.765 |

## 3. 解读

1. **3/5 分钟不是不可做**：事件级 `state@3` 仍有 93.5%（TT）/96.4%（OB）的根因信号，
   `state@5` 有 85.7%/85.5%。说明即使提前 5 分钟，多数事件的根因已经在当前窗口里可观测。
2. **窗口级会掉，但事件级稳**：`state@5` 的窗口信号比例只有 70.4%/64.4%，
   因为很多 5 分钟正例窗口是故障的“持续/衰减”段，根因信号已回落；
   但每个事件至少有一个窗口有信号，所以事件级仍高。
3. **first-visible 也可做**：TT 的 3/5 分钟事件信号 92.5%，OB 76.5%；
   OB 偏低是因为它的入口 operation 少、故障前兆更弱。
4. **结论**：可以进入多 horizon 的预测与 CPCA；但评价应以**事件级**为主，
   窗口级 5 分钟会低估可归因性。

## 4. 下一步

1. 训练 `state@3/5` 的预测器（horizon-specific），对比 AP/F1；
2. CPCA 加 trailing window（$w=\max(1,h)$），在 3/5 分钟上重算事件级排序；
3. 若 3/5 的 CPCA 明显低于 1 分钟，按“事件级信号仍在、但窗口级衰减”解释，
   不强行承诺窗口级指标。

## 5. 复现

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/gate_horizon_leadtime.py \
  --dataset tt --horizons 1 3 5 --output artifacts/gate_horizon_tt_v1.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/gate_horizon_leadtime.py \
  --dataset ob --horizons 1 3 5 --output artifacts/gate_horizon_ob_v1.json
```
