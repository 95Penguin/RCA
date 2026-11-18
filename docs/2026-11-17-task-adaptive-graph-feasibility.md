# 任务自适应图交互最小可行性实验

结论：**NO-GO for the current task-independent graph-expert design**。

本实验只验证一个最小假设：故障预测与根因定位是否因共享同一套跨服务图传播参数而受到限制。它不验证传播方向，也不引入门控、额外损失或动态图学习。

## 实验设计

固定条件：

- RE2-OB 与 RE2-TT fair panel；
- exact next-minute state task；
- deployment split fold 0；
- seed 42，30 epochs；
- 相同历史窗口、模态编码器、预测头、RCA 头和损失；
- `node_loss_weight=0`，`rank_loss_weight=0.5`。

模型：

| 代号 | 模型 | 图传播 |
|---|---|---|
| A | `ltfp` | 两个任务共享一个两层 GAT |
| B | `ltfp-task-graph` | 预测和 RCA 分别使用独立的两层 GAT |
| C | `ltfp-shared-graph-experts` | 两个 GAT 取平均，两个任务使用同一个平均表示 |
| D | `ltfp-no-graph` | 不进行跨服务消息传播 |

B 与 C 的参数量完全相同，均为 147,271；A 为 138,567，D 为 129,863。B/C 之间的区别仅是两个任务是否使用不同的专家结果。

## 第零步：共享图层梯度冲突

在现有 fold-0 LTFP checkpoint 的训练窗口上，计算 failure loss 与 root ranking loss 对共享 GAT 参数的梯度余弦。RCA 梯度只在具有根因标签的正例窗口上计算。

| 数据集 | 正例 | mixed-batch mean cosine | mixed aggregate cosine | positive-only mean cosine | positive aggregate cosine |
|---|---:|---:|---:|---:|---:|
| OB | 200 | +0.087 | -0.041 | +0.063 | +0.055 |
| TT | 280 | +0.061 | +0.065 | +0.032 | -0.033 |

两套统计均接近 0，批次内正负方向混合，没有出现跨数据集稳定的强负冲突。这不能证明共享图最优，但没有为拆分 GAT 提供强先验证据。

## 30-epoch 筛选结果

| 数据集 | 模型 | AP | F1 | Hit@1 | MRR |
|---|---|---:|---:|---:|---:|
| OB | A Shared LTFP | 0.893 | 0.737 | 0.889 | 0.931 |
| OB | B Task-independent | 0.887 | 0.667 | 0.833 | 0.880 |
| OB | C Shared experts | 0.831 | 0.669 | 1.000 | 1.000 |
| OB | D No graph | 0.857 | 0.706 | 0.722 | 0.794 |
| TT | A Shared LTFP | 0.622 | 0.562 | 0.533 | 0.700 |
| TT | B Task-independent | 0.570 | 0.569 | 0.667 | 0.789 |
| TT | C Shared experts | 0.609 | 0.576 | 0.533 | 0.711 |
| TT | D No graph | 0.557 | 0.595 | 0.667 | 0.740 |

关键差值（B 相对 A）：

| 数据集 | ΔAP | ΔF1 | ΔHit@1 | ΔMRR |
|---|---:|---:|---:|---:|
| OB | -0.005 | -0.070 | -0.056 | -0.051 |
| TT | -0.052 | +0.008 | +0.133 | +0.089 |

## 结论

1. **OB 不支持任务独立图传播。** B 的 AP 略降，F1、Hit@1、MRR 均明显下降。
2. **TT 出现预测—定位权衡，而不是双赢。** B 改善 Hit@1/MRR，但 AP 下降 0.052，超过预设的可接受退化范围。
3. **额外容量不是统一解释。** C 在 TT 接近 A；在 OB 则显著降低 AP、提高 RCA。两个专家更像改变多任务优化平衡，而不是稳定增加能力。
4. **图传播对预测总体有价值。** D 在 OB/TT 的 AP 分别比 A 低 0.036/0.065；但单个 fold 的 RCA 变化不一致，不能据此主张任务需要不同图。
5. **不进入完整模型阶段。** 不继续增加任务门控、双向传播或差异正则，也不进行五折多种子扩展。

本结论的边界是：它否定当前“完全独立 GAT 专家”作为同时提升 AP 和 MRR 的候选设计，不等于所有任务自适应图方法都不可能有效。当前证据更支持先研究输入图本身的可靠性，而不是继续拆分任务参数。

## 可复现入口

- 梯度审计：`scripts/audit_ltfp_graph_gradient_conflict.py`
- A/B/C/D 实验：`scripts/run_task_graph_experiment.py`
- 模型实现：`src/models/ltfp.py`
- 模型注册：`src/models/factory.py`

本次原始筛选输出位于 `/private/tmp/task_graph_pilot_30ep`；结论性数值已记录在本文档中。
