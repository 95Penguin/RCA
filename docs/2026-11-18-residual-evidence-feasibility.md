# 安全残差服务证据最小可行性实验

结论：**局部成立，但未通过跨数据集继续门槛**。

## 为什么重新实验

此前 Null-Joint 强制同一个标量服务证据同时承担系统预测和 RCA，形成了过强瓶颈。本实验保留原始 LTFP 全局故障头，只将服务证据作为零初始化残差：

`final_logit = global_logit + alpha * evidence_log_odds`

服务证据分支在所有窗口上接受 normal/failure 辅助监督，Joint 额外接受根因排序监督。Marginal 与 Joint 架构和参数量完全相同。

所有模型统一按 validation AP 选择 checkpoint，消除上一轮不同损失导致的选择偏差。

## 协议

- RE2 fair panel；
- state 1-minute；
- deployment fold 0；
- seed 42，30 epochs；
- evidence loss weight 0.1；
- root loss weight 0、0.05、0.1；
- validation AP checkpoint selection。

## 结果

| 数据集 | 模型 | Root λ | AP | F1 | Hit@1 | MRR | alpha |
|---|---|---:|---:|---:|---:|---:|---:|
| OB | LTFP | 0.5 | 0.836 | 0.606 | 0.778 | 0.880 | — |
| OB | Residual-Marginal | 0 | 0.869 | 0.730 | 0.444 | 0.561 | 0.079 |
| OB | Residual-Joint | 0.1 | 0.888 | 0.700 | 0.944 | 0.972 | 0.079 |
| OB | Residual-Joint | 0.05 | 0.931 | 0.839 | 0.500 | 0.671 | 0.082 |
| TT | LTFP | 0.5 | 0.618 | 0.585 | 0.533 | 0.715 | — |
| TT | Residual-Marginal | 0 | 0.592 | 0.550 | 0.267 | 0.401 | 0.071 |
| TT | Residual-Joint | 0.1 | 0.560 | 0.548 | 0.533 | 0.684 | 0.072 |
| TT | Residual-Joint | 0.05 | 0.610 | 0.583 | 0.467 | 0.627 | 0.125 |

## 核心对照

Joint(0.1) 相对 Marginal：

| 数据集 | ΔAP | ΔHit@1 | ΔMRR |
|---|---:|---:|---:|
| OB | +0.019 | +0.500 | +0.411 |
| TT | -0.032 | +0.267 | +0.282 |

根因监督在两个数据集上都显著改善服务排名，但只在 OB 上同时改善 AP；TT 出现明显负迁移。

Joint 相对同选模 LTFP：

| 数据集 | Root λ | ΔAP | ΔHit@1 | ΔMRR |
|---|---:|---:|---:|---:|
| OB | 0.1 | +0.051 | +0.167 | +0.093 |
| TT | 0.1 | -0.058 | +0.000 | -0.031 |
| OB | 0.05 | +0.095 | -0.278 | -0.208 |
| TT | 0.05 | -0.008 | -0.067 | -0.088 |

降低根因权重能够恢复 AP，但会失去 RCA。不存在一个已测权重在 OB、TT 上同时稳定改善 AP 与 MRR。

## 判断

1. **此前框架确实过强。** 安全残差版本在 OB 的 λ=0.1 下同时提升 AP、Hit@1、MRR，证明原始想法并非完全无效。
2. **服务证据没有被自动忽略。** 所有残差模型均学到正的 alpha（0.071–0.125）。
3. **根因监督的预测收益不具跨数据集稳定性。** OB 增益，TT 负迁移。
4. **当前问题不是再增加门控即可解决。** 不同 λ 只是在 AP/MRR Pareto 前沿上移动，没有产生统一双赢点。
5. **停止扩展。** 不进行五折、多种子和样本级可靠性门控，避免在单数据集正结果上继续调参。

## 可复现入口

- 模型：`src/models/coarse_fine.py::LTFPResidualEvidence`
- 训练支持：`src/training/engine.py`、`src/training/experiment.py`
- 实验脚本：`scripts/run_residual_evidence_experiment.py`
- 原始输出：`/private/tmp/residual_evidence_pilot_30ep/`
