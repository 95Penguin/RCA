# 服务局部状态与 Trace 关系分开保留：第一阶段验证

## 范围与协议

这是第一阶段 RCA 机制验证。系统级预测沿用冻结的历史 LTFP；尚未实现服务证据向系统预测反馈，不能用本轮 AP 不变宣称跨粒度协作有效。

RE2-OB / RE2-TT，deployment split 五折，历史 5 分钟，预测下一分钟状态。基线使用 `re2_{dataset}_ltfp_deployment_fold{fold}_fair_30ep_seed42_baseline_v1/best_model.pt`；这与正常偏差审计的历史 checkpoint 相同，不能和后续重训并使用其他 checkpoint 选择规则的结果混用。

三个配置：

1. A：冻结原 LTFP 系统预测与监督式 RCA。
2. B（Local）：Metric 历史、正常偏差和观测 mask 经独立 TCN；图层只使用自连接。
3. C（State–Relation）：与 B 完全相同的参数和初始化，图层使用现有 Trace 无向调用次数图。

新增分支不用 Logs / Trace 的节点特征，但冻结的 LTFP 基线仍使用原三模态输入。因此本实验并没有验证“完整新模型可以删掉日志”。

## 模型流程

训练集全正常窗口 → 每个服务、历史位置、指标的 median / MAD 原型。

`Metric values + normal deviations + masks → independent TCN → local state`

同时将偏差 latest / history-max / history-mean 投影，与 TCN 输出相加并规范化。缺失指标不贡献偏差；无观测的原型单元退回 median=0、scale=1。

`local state → one graph attention layer → relation state`

将 `[local, relation, local * relation]` 输入 MLP，输出 RCA logit 修正：

`new RCA logits = frozen LTFP RCA logits + learned residual`

修正头最后一层权重和偏置全零，所以初始化时两个输出都严格恢复基线。B 与 C 的图层参数量相同，自连接对照仍能进行逐服务变换，排除简单容量增加的影响。

## 训练与评估

- 原 LTFP 所有参数冻结，dropout 与 BatchNorm 均保持 eval。
- 每折 sidecar seed=42，30 轮，AdamW lr=0.001，batch size=64。
- 仅在未来故障且有根因标签的窗口计算 RCA softmax 排序损失；按事件长度平衡权重。
- 两个分支使用相同 minibatch 顺序；新增分支 dropout=0，仅独立 Metric TCN 的 BatchNorm 在训练中更新。
- 按验证集事件 MRR 选择 checkpoint，同分取最早轮次。测试指标不参与选择。
- 事件级评估对同一事件的可用窗口分数取均值，是回顾性汇总，不代表故障开始之前就定位正确。脚本同时记录窗口级指标。
- 缓存冻结主干 logits 加速训练，结束后用完整模型复算并检查 RCA 一致；基线参数冻结保障系统预测。

## 实验结果

结果和逐轮验证轨迹见 [JSON 汇总](../artifacts/state_relation_pilot_2026-11-18.json)。五折非加权均值：

| 数据集 | 配置 | AP | Hit@1 | MRR |
| --- | --- | ---: | ---: | ---: |
| OB | A：历史 LTFP | 0.918803 | 0.807009 | 0.878226 |
| OB | B：Local | 0.918803 | 0.807009 | 0.880541 |
| OB | C：State–Relation | 0.918803 | 0.853675 | 0.901414 |
| TT | A：历史 LTFP | 0.845770 | 0.894167 | 0.931667 |
| TT | B：Local | 0.845770 | 0.894167 | 0.932111 |
| TT | C：State–Relation | 0.845770 | 0.894167 | 0.932111 |

事件 MRR 逐折结果：

| 数据集 | fold | A | B | C |
| --- | ---: | ---: | ---: | ---: |
| OB | 0 | 0.930556 | 0.935185 | 0.935185 |
| OB | 1 | 0.923077 | 0.923077 | 0.923077 |
| OB | 2 | 1.000000 | 1.000000 | 1.000000 |
| OB | 3 | 0.687500 | 0.694444 | 0.857143 |
| OB | 4 | 0.850000 | 0.850000 | 0.791667 |
| TT | 0 | 0.700000 | 0.702222 | 0.702222 |
| TT | 1 | 1.000000 | 1.000000 | 1.000000 |
| TT | 2 | 1.000000 | 1.000000 | 1.000000 |
| TT | 3 | 0.958333 | 0.958333 | 0.958333 |
| TT | 4 | 1.000000 | 1.000000 | 1.000000 |

两组分支的可训练参数均为 51,970。AP 不变是冻结结构带来的确定性保障，不属于性能提升。

## 判断与下一步

后续已完成 [事件复查与拓扑对照](2026-11-18-state-relation-graph-audit.md)：错配图重训能够重现 fold 3 的主要改善，因此不能把本表的收益归因于真实 Trace 拓扑。

- OB 的 C 相比 A，MRR 增加 0.023188，Hit@1 增加 0.046667；但主要收益来自 fold 3，fold 4 出现 0.058333 的 MRR 退化。尚不满足稳定跨折改善的条件。
- TT 只有 fold 0 一个较低名次发生改善，五折 MRR 增加 0.000444，Hit@1 不变，实际效果接近持平。C 没有优于参数匹配的 B。
- B 在全部折中大多选择第 1 轮；C 在 OB fold 3 选择第 8 轮，fold 4 选择第 19 轮。需要区分关系本身的收益和验证选模不稳定问题，不能用测试 MRR 重新挑轮次。
- 本轮只支持“Trace 关系在部分部署中能帮助局部证据分支”，尚未支持通用的状态—关系增益，也未验证服务证据能提高系统 AP。

建议先分析 OB fold 3 修正了哪些服务、fold 4 损害了哪些服务，并补随机/打乱边控制，确认收益确实依赖真实调用关系。在重复种子和边控制确认前，不扩展有向多通道图或双向任务交互。

完整 checkpoint 位于 `/private/tmp/state_relation_cached_fivefold_20261118`；JSON 汇总已保存到仓库 artifacts。56 项测试通过，包括正常原型训练范围、缺失观测屏蔽、零初始化、容量匹配、梯度隔离和缓存/完整模型输出一致性。

## 解释边界

即使 C 优于 B，也只能支持“在固定 LTFP 基础上，新增服务状态分支能从 Trace 调用关系获得 RCA 收益”。不能据此声称恢复真实故障传播或因果关系。

此前 raw metric deviation 强于 latent deviation，只说明简单距离度量在两种表示上的根因可分性不同，不能单凭它证明编码器已经丢失信息。监督式 LTFP RCA 本身仍强于 raw deviation 的五折平均值。

本实验也没有完全分开正常原型、TCN、状态跳连和新增参数各自的贡献；若有稳定收益，需要再补去掉 deviation、图后单一表示等消融。

## 复现

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B \
  scripts/run_state_relation_pilot.py \
  --datasets ob tt --folds 0 1 2 3 4 --epochs 30 --seed 42 \
  --device cpu --output-dir /private/tmp/state_relation_new_run
```

每折保存两份 checkpoint；`results.json` 保存 AP、事件与窗口 RCA、原型样本量、选中轮次和训练轨迹。输出目录需使用新目录。
