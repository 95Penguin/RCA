# 根因梯度保护最小实验

## Review 结论

原计划中的独立 Adapter、异常/根因双头、表示层交互和可靠性门控都有合理动机，但不能一次加入。一次改变多个机制后，即使指标上升，也无法判断收益来自哪里。

本轮只验证一个失败机制：

> Residual-Joint 在 TT 上的 AP 下降，是否主要因为根因排序损失修改了粗粒度预测依赖的共享时序/GAT 表示？

## 严格对照设计

新增 `ltfp-residual-protected-joint`：

- 与 `ltfp-residual-joint` 参数量完全相同；
- 三模态输入、前向预测、RCA 输出和损失权重相同；
- 推理时没有新增路径；
- 唯一差别：`L_root` 使用基于 detached 服务表示重新计算的排序 logits。

因此根因排序损失仍可更新服务证据头，但不能更新：

- 模态时序编码器；
- 模态融合层；
- GAT；
- 系统级池化与故障头。

故障预测损失和服务证据二分类损失仍按原路径训练，不受影响。

## 协议

- RE2-OB、RE2-TT；
- exact next-minute state；
- deployment fold 0；
- Metrics + Logs + Traces；
- seed 42，30 epochs；
- evidence loss weight `0.1`；
- root loss weight `0.1`；
- validation AP checkpoint selection。

## 结果

**口径提醒：本表只比较 deployment fold 0，不是此前组会报告中的五折均值。** 组会报告中 TT-LTFP 的 `AP=0.846、MRR=0.932` 是 folds 0–4 的非加权均值；其中 fold 0 本身明显更难。

| TT-LTFP 口径 | AP | F1 | Hit@1 | MRR |
|---|---:|---:|---:|---:|
| 旧报告五折均值 | 0.846 | 0.771 | 0.894 | 0.932 |
| 旧报告 fold 0（validation loss 选模） | 0.622 | 0.562 | 0.533 | 0.700 |
| 本轮 fold 0（validation AP 选模） | 0.618 | 0.585 | 0.533 | 0.715 |

旧报告五折的 TT-LTFP AP 分别为 `0.622/0.939/0.900/0.887/0.881`，MRR 分别为 `0.700/1.000/1.000/0.958/1.000`。因此，fold 0 适合做低成本困难场景筛选，但不能替代五折最终结论。

| 数据集 | 模型 | AP | F1 | Hit@1 | MRR |
|---|---|---:|---:|---:|---:|
| OB | LTFP | 0.836 | 0.606 | 0.778 | 0.880 |
| OB | Residual-Marginal | 0.869 | 0.730 | 0.444 | 0.561 |
| OB | Residual-Joint | 0.888 | 0.700 | 0.944 | 0.972 |
| OB | **Protected-Joint** | **0.951** | **0.820** | 0.667 | 0.729 |
| TT | LTFP | 0.618 | 0.585 | 0.533 | 0.715 |
| TT | Residual-Marginal | 0.592 | 0.550 | 0.267 | 0.401 |
| TT | Residual-Joint | 0.560 | 0.548 | 0.533 | 0.684 |
| TT | **Protected-Joint** | **0.624** | 0.565 | 0.400 | 0.601 |

分支诊断：

| 数据集 | Protected 最终 AP | 粗粒度分支 AP | 服务证据分支 AP | alpha |
|---|---:|---:|---:|---:|
| OB | 0.951 | 0.950 | 0.956 | 0.054 |
| TT | 0.624 | 0.621 | 0.605 | 0.134 |

## 判断

### 1. 共享主干中的根因梯度冲突得到支持

在 fold 0 上，TT 的 AP 从原 Joint 的 `0.560` 恢复到 `0.624`，回到原始 LTFP 的约 `0.618–0.622`；这不是相对五折 LTFP `0.846` 的提升。OB fold 0 的 AP 则从 `0.888` 上升到 `0.951`。

这支持“根因排序梯度进入共享主干是 fold 0 预测负迁移的重要来源”，但单折结果还不能证明该机制在所有配置上成立。最终 AP 与粗粒度分支 AP 非常接近，也说明 fold 0 的恢复主要发生在表示层，而不是靠最后的残差偶然修正。

### 2. 完全阻断不是最终答案

Protected-Joint 的 MRR 在 OB/TT 分别为 `0.729/0.601`，低于原 Joint 的 `0.972/0.684`，也低于 LTFP 的 `0.880/0.715`。

根因监督不仅会干扰系统预测，也确实需要改变一部分服务级表示才能提升 RCA。完全共享偏向 RCA、完全阻断偏向 AP，两者是两个极端。

### 3. 下一步不应直接加入可靠性门控

当前失败发生在表示和梯度层，门控只能调整最终融合比例，不能给 RCA 提供可适配、又不会污染粗粒度分支的表示空间。

## 下一步最小设计

下一轮应验证“局部可塑、全局受保护”：

1. 共享编码器输出基础服务表示；
2. 粗粒度分支使用独立 coarse adapter；
3. 细粒度分支使用独立 fine adapter；
4. `L_root` 可更新 fine adapter，但不能更新 coarse adapter；
5. 将异常证据头 `a_i` 与根因责任头 `r_i` 分开；
6. 暂时仍使用零初始化表示残差，不加入样本级门控。

还需要一个容量匹配对照，避免把 Adapter 增益错误解释成任务解耦增益。

继续门槛应预先固定：相对 LTFP，OB 和 TT 的 AP 均不下降超过 `0.01`，并且两个数据集的 MRR 均有提升，才进入五折多种子实验。

## 实现与原始输出

- 模型：`src/models/coarse_fine.py::LTFPProtectedRootEvidence`
- 根因损失路由：`src/training/engine.py`
- 实验入口：`scripts/run_residual_evidence_experiment.py`
- 临时原始输出：`/private/tmp/residual_protected_pilot_30ep/`
