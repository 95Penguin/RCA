# OC-SID Gate A：Operation 条件化 Trace 路径证据审计

日期：2026-11-23

## 目的

在开发完整 OC-SID 前，检验其核心数据假设：请求级 operation 路径上的 inclusive/exclusive latency，是否比当前每服务全局平均 span duration 更能区分根因与传播症状。

## 实现边界

- 从 RE2 原始 `traces.csv` 恢复 `traceID/spanID/parentSpanID` 请求 DAG。
- 每个 span 只在完成后可见；输入分钟之后完成的 span 不参与当前预测。
- exclusive latency 从父 span 区间中扣除直接子 span 时间区间的并集，避免并发子调用重复扣除。
- 未作为根因候选的数据库/基础设施子 span 仍参与等待时间扣除。
- 每个 case 的十个完整注入前分钟构造正常参考；因此本实验是数据可行性上界审计，正式在线模型必须改成训练期或纯过去滚动基线。
- TT 从当前输入分钟选择 entry-p95 正向 robust deviation 最大且请求数充分的 operation；不使用下一分钟 operation 标签。
- 主比较预先固定为 `operation_exclusive` 对 `global_mean_duration`。
- 结果出来后增加的固定 robust-z 求和只作为互补性探索，不能冒充预注册结果。

## 主结果

### RE2-OB

| 方法 | Window Hit@1 / MRR | Event Hit@1 / MRR | 首分钟 Hit@1 / MRR |
|---|---:|---:|---:|
| 全局平均 duration | 0.235 / 0.521 | 0.345 / 0.588 | 0.327 / 0.542 |
| Operation inclusive | 0.275 / 0.464 | 0.382 / 0.543 | 0.273 / 0.428 |
| Operation exclusive | **0.321** / 0.484 | **0.400** / 0.550 | 0.309 / 0.444 |

根因位于所选 operation 路径的窗口比例为 99.1%。Exclusive 能将更多根因推到第一名，但中后部排序明显变差，事件和首分钟 MRR 均低于全局 duration。OB 主 Gate 不通过。

### RE2-TT

| 方法 | Window Hit@1 / MRR | Event Hit@1 / MRR | 首分钟 Hit@1 / MRR |
|---|---:|---:|---:|
| 全局平均 duration | 0.617 / 0.779 | 0.719 / 0.833 | 0.500 / 0.679 |
| Operation inclusive | **0.706 / 0.801** | **0.813 / 0.890** | 0.516 / 0.616 |
| Operation exclusive | 0.690 / 0.797 | 0.719 / 0.839 | **0.578** / 0.657 |

根因位于所选 operation 路径的窗口比例为 90.4%，首分钟只有 71.9%。Operation inclusive 对 TT 整体事件定位明显有效；exclusive 提高首分钟 Hit@1，但仍降低首分钟 MRR。Exclusive 作为普适主分数的 Gate 仍不通过。

## 按故障类型的证据

TT 的 Event Hit@1 / MRR：

| 故障 | 全局 duration | Inclusive | Exclusive |
|---|---:|---:|---:|
| CPU | 0.727 / 0.826 | 0.909 / 0.939 | **0.909 / 0.955** |
| Delay | 0.818 / 0.894 | **0.909 / 0.955** | 0.455 / 0.727 |
| Disk | 0.889 / 0.917 | **0.889 / 0.926** | 0.778 / 0.870 |
| Loss | 0.636 / 0.803 | 0.455 / 0.697 | **0.909 / 0.927** |
| Memory | 0.538 / 0.718 | **0.769 / 0.856** | 0.538 / 0.724 |
| Socket | 0.778 / 0.889 | **1.000 / 1.000** | 0.778 / 0.861 |

Inclusive 和 exclusive 不是谁绝对优于谁，而是描述不同故障表现：delay/memory/socket 更常表现为路径累计时间，CPU/loss 的局部 exclusive 变化更有判别力。固定使用某一种证据与数据不符。

## 探索性互补结果

在相同 capped robust-z 尺度上直接相加，不训练权重：

| 数据集 | 方法 | Event Hit@1 / MRR | 首分钟 Hit@1 / MRR |
|---|---|---:|---:|
| OB | 全局 duration | 0.345 / 0.588 | 0.327 / 0.542 |
| OB | 全局 + inclusive + exclusive | **0.436 / 0.591** | 0.309 / 0.482 |
| TT | 全局 duration | 0.719 / 0.833 | 0.500 / 0.679 |
| TT | 全局 + inclusive + exclusive | **0.875 / 0.926** | **0.719 / 0.809** |

这说明 operation 路径证据不是全局服务证据的替代品，而是互补证据。TT 的互补性很强，OB 只改善事件 Hit@1，首分钟反而受损。

## 结论与方法修正

原始命题“exclusive latency 可普适区分根因与症状”被否定。完整 OC-SID 不应强制将根因分数定义为 exclusive source score。

保留且有数据支持的命题是：

1. 全局服务时延、operation inclusive 和 operation exclusive 提供互补信息。
2. Operation 路径选择对 TT 很有效，但首分钟存在明显路径覆盖缺口。
3. 后续责任模型需要学习“何时信任路径证据”，且必须允许退回全局证据。

因此后续结构调整为：

- 下一分钟 operation risk 产生路径证据可靠性门控；
- 全局服务证据作为永不移除的基础排序；
- inclusive/exclusive 只作为有界残差；
- 根因不在所选路径或 operation 置信度低时，残差严格趋近 0；
- normal-restoration responsibility 在这个受保护的组合上计算，而不是在纯 exclusive 分数上计算。

## 复现

```bash
PYTHONDONTWRITEBYTECODE=1 baselines/.venv/bin/python \
  scripts/audit_operation_source_impact.py \
  --output artifacts/re2_operation_source_impact_gate_a_v1.json
```
