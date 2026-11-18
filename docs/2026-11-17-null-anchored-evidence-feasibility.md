# 无故障锚定服务证据最小可行性实验

结论：**NO-GO for the null-anchored single-evidence design**。

## 核心假设

根因标签提供比系统故障标签更细粒度的监督，可能帮助同一组服务级证据同时获得：

1. 服务区分能力；
2. 系统故障判别能力。

模型为每个服务产生证据 `e_i`，并产生无故障证据 `e_0`。系统故障 logit 使用服务证据的 size-normalized log-mean-exp 与 `e_0` 比较，RCA 使用相同 `e_i` 的条件 softmax。

## 严格对照

`Null-Joint` 与 `Null-Marginal` 的架构、参数量、初始化、数据、故障损失和训练预算完全相同。

- `Null-Joint`：`rank_loss_weight=0.5`，使用根因细粒度监督；
- `Null-Marginal`：`rank_loss_weight=0`，完全不使用根因监督。

固定协议：RE2 fair panel、state 1-minute、deployment fold 0、seed 42、30 epochs。原始 LTFP 与 RC² 使用已有同协议 checkpoint。

## 结果

| 数据集 | 模型 | AP | F1 | Hit@1 | MRR |
|---|---|---:|---:|---:|---:|
| OB | LTFP | 0.893 | 0.737 | 0.889 | 0.931 |
| OB | RC² | 0.817 | 0.605 | 0.944 | 0.972 |
| OB | Null-Joint | 0.808 | 0.605 | 1.000 | 1.000 |
| OB | Null-Marginal | 0.809 | 0.587 | 0.333 | 0.495 |
| TT | LTFP | 0.622 | 0.562 | 0.533 | 0.700 |
| TT | RC² | 0.537 | 0.535 | 0.600 | 0.715 |
| TT | Null-Joint | 0.529 | 0.574 | 0.467 | 0.630 |
| TT | Null-Marginal | 0.625 | 0.574 | 0.467 | 0.597 |

核心因果对照（Joint 相对 Marginal）：

| 数据集 | ΔAP | ΔF1 | ΔHit@1 | ΔMRR |
|---|---:|---:|---:|---:|
| OB | -0.002 | +0.018 | +0.667 | +0.505 |
| TT | -0.096 | +0.000 | +0.000 | +0.033 |

Joint 相对原始 LTFP：

| 数据集 | ΔAP | ΔF1 | ΔHit@1 | ΔMRR |
|---|---:|---:|---:|---:|
| OB | -0.085 | -0.132 | +0.111 | +0.069 |
| TT | -0.094 | +0.012 | -0.067 | -0.070 |

## 判断

1. **根因监督确实能够学习服务区分。** OB 中 MRR 从 0.495 提升至 1.000。
2. **服务区分没有转化为系统判别能力。** OB AP 不变；TT AP 下降 0.096。
3. **统一标量证据形成过强瓶颈。** 相对根因排序和绝对系统风险需要不同自由度。把两者压入同一 `e_i` 会重新产生预测—定位权衡。
4. **无故障锚点没有修复 RC² 的主要问题。** Null-Joint 与 RC² 都明显低于原始 LTFP 的 AP。
5. **不继续增加可靠性门控。** 当前服务证据本身没有同时具备系统判别能力；在其上增加门控只会改变使用方式，不能证明细粒度监督改善了证据。

本结果只否定当前“同一标量服务证据同时承担故障预测与 RCA”的实现，不否定根因标签作为辅助监督的一切可能性。但结合已有 fault-type auxiliary、dual-router、RC² 和 task-independent GAT 的负结果，继续增加任务交互模块的优先级很低。

## 可复现入口

- 模型：`src/models/ltfp.py::LTFPNullAnchoredEvidence`
- 模型注册：`src/models/factory.py`
- 实验脚本：`scripts/run_null_evidence_experiment.py`
- 原始筛选输出：`/private/tmp/null_evidence_pilot_30ep/null_evidence_summary.json`
