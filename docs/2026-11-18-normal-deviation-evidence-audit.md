# 正常原型偏差证据五折审计

## 结论

“系统级粗粒度 + 服务级细粒度”假设仍然值得继续，但现有失败主要来自细粒度证据的构造方式，而不是任务本身没有局部信号。

训练集正常窗口形成的**服务级 metric 正常原型偏差**，在两个数据集的五个 deployment folds 中都显著优于从 LTFP 隐表示计算的偏差。说明原始 metric 中存在稳定的根因区分信息，而共享时序编码、模态融合和图传播会丢失或混合其中一部分信息。

它尚未整体超过有根因监督的 LTFP RCA head，因此不能直接替换监督式 RCA。更合理的用途是：保留一条不经过共享编码和图传播的 metric deviation side path，为系统级预测和监督式 RCA 提供互补局部证据。

## 验证问题

只验证一个假设：

> 根因服务相对自身正常状态的偏差，是否比非根因服务更明显；这种证据是否在 LTFP 编码前比编码后更容易分辨？

本实验不训练新模型，也不使用测试标签拟合原型或参数。

## 协议

- 数据：RE2-OB、RE2-TT，deployment split 五折。
- 历史窗口：5 分钟；目标：下一分钟根因服务。
- 正常参考：仅使用训练折中“完整历史及下一时刻均正常”的窗口。
- 正常样本量：OB 每折 539–649 个；TT 每折 544–582 个。
- 原型：按历史位置、服务位置、特征维度计算训练集 median。
- 尺度：MAD；退化维度使用标准差，再退化则使用 1。
- 偏差：`clip(abs((x - median) / scale), 0, 10)`。
- 事件聚合：同一故障事件内窗口分数取平均，再计算 Hit@K/MRR。
- 对照：历史 LTFP checkpoint 的监督式 RCA head、LTFP local state 偏差、图传播后 node state 偏差。

## 五折结果

表中为五折非加权均值；MRR 后括号为五折总体标准差。

### RE2-OB

| 证据 | Hit@1 | MRR |
| --- | ---: | ---: |
| LTFP 监督式 RCA head | 0.807 | 0.878 (0.107) |
| raw metric latest deviation | 0.682 | 0.757 (0.099) |
| raw metric history-max deviation | 0.682 | 0.758 (0.098) |
| raw all-modality latest deviation | 0.249 | 0.566 (0.125) |
| LTFP local-state deviation | 0.308 | 0.547 (0.173) |
| LTFP graph-state deviation | 0.423 | 0.665 (0.072) |

### RE2-TT

| 证据 | Hit@1 | MRR |
| --- | ---: | ---: |
| LTFP 监督式 RCA head | 0.894 | 0.932 (0.117) |
| raw metric latest deviation | 0.841 | 0.892 (0.088) |
| raw metric history-max deviation | 0.831 | 0.880 (0.093) |
| raw all-modality latest deviation | 0.779 | 0.861 (0.071) |
| LTFP local-state deviation | 0.353 | 0.580 (0.106) |
| LTFP graph-state deviation | 0.482 | 0.649 (0.124) |

## 折间互补性

事件级 MRR（LTFP 监督式 RCA / raw metric latest）：

- OB：fold 0 `0.931 / 0.671`，fold 1 `0.923 / 0.676`，fold 2 `1.000 / 0.687`，fold 3 `0.688 / 0.854`，fold 4 `0.850 / 0.900`。
- TT：fold 0 `0.700 / 0.784`，fold 1 `1.000 / 0.913`，fold 2 `1.000 / 0.795`，fold 3 `0.958 / 0.969`，fold 4 `1.000 / 1.000`。

偏差证据不是无条件优于监督头，但它在 LTFP 较弱的 OB folds 3/4 和 TT fold 0 上提供了补偿。这正是适合做“受保护的局部证据分支”的情形；直接替换或固定权重硬融合都会损伤原本已经正确的折。

## 模态结论

latest deviation 的五折 MRR：

| 数据集 | Metrics | Logs | Traces | 全模态直接平均 |
| --- | ---: | ---: | ---: | ---: |
| OB | 0.757 | 0.299 | 0.363 | 0.566 |
| TT | 0.892 | 0.310 | 0.625 | 0.861 |

因此不能把三种模态等权作为“局部证据”。Metrics 是主要载体；trace 在 TT 有辅助价值；logs 的直接偏差分数接近噪声。原先的自由 evidence MLP 和模态混合可能把弱模态噪声引入了 RCA 表示。

## 对此前失败的解释

1. 细粒度分支从共享 LTFP 表示出发，局部信息已经被模态融合和任务目标压缩。
2. 在提取根因证据前做图传播，会把根因异常扩散到症状服务，使正常偏差不再局部。
3. 自由 MLP 没有“正常状态”锚点，容易记住训练部署中的根因—症状共现关系。
4. 可靠性门没有独立监督，先前实验中几乎总是饱和到 1，无法拒绝错误证据。
5. 三模态直接融合不成立；OB 上 all-modality MRR 比 metric-only 低 0.191。
6. 验证部署和测试部署的根因服务分布差异较大，单纯按验证 MRR 选择交互模型或 checkpoint 不稳定。

## 下一版最小模型

先实现 `LTFP + Protected Metric Deviation Branch`，不加入新的 GAT、复杂门控或三模态等权融合：

1. 保留原 LTFP 系统级预测路径和监督式 RCA head。
2. 用训练正常窗口建立每个服务/指标的 robust prototype。
3. 原始 metrics 直接生成服务级 deviation vector，不经过共享编码器和图传播。
4. deviation branch 单独输出根因分数；LTFP 监督 RCA head 继续作为主分支。
5. 第一轮只比较三种预先固定的组合：原 LTFP、deviation-only、logit residual fusion。
6. 融合必须采用“零初始化残差 + 对原 LTFP 的保护约束”，保证初始行为严格等于基线，并让模型只学习何时修正弱折。
7. 系统 AP 暂时完全沿用 LTFP，不允许 RCA 梯度进入系统预测主干；先验证 MRR/Hit@1 能否提高且 AP 不变。

停止条件：如果受保护残差融合在 OB、TT 五折均值上不能提高 MRR，或只靠某一折拉高均值，就不再扩展该分支。

## 复现

```bash
.venv/bin/python -B scripts/audit_normal_deviation_evidence.py \
  --dataset ob --folds 0 1 2 3 4 --device auto \
  --output artifacts/normal_deviation_evidence_ob.json

.venv/bin/python -B scripts/audit_normal_deviation_evidence.py \
  --dataset tt --folds 0 1 2 3 4 --device auto \
  --output artifacts/normal_deviation_evidence_tt.json
```

