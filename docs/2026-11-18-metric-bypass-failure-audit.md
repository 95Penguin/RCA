# Metric 旁路退化事件分析与阶段决策

本轮只重放已选 checkpoint，分析 OB fold 0 / TT fold 3、sidecar seeds 42 和 43；没有训练新参数，没有重新选模，也没有用测试标签修改模型。

## 为什么验证，以及何时停止

原型偏差审计确认存在可利用的局部信号；错配图对照排除了“改善必然来自正确调用关系”的解释；原始指标与显式偏差对照表明，部分收益来自指标旁路本身，偏差不是唯一原因。

这些验证服务于方法选择，不应无限延长。本轮已有足够依据做出决策：暂停增加编码/图模块，把下一版集中在局部证据修正的行为约束。每次结构修改都应对应一个明确失败机制，并只进行必要对照。

## OB fold 0：验证根因与测试根因完全不同

有效根因事件覆盖：

- 训练：productcatalog 10、recommendation 6、currency 4、email 2、checkout 2。
- 验证：productcatalog 7、recommendation 6。
- 测试：checkout 5、currency 13。

测试根因在训练中出现过，但验证根因服务集合与测试完全不相交。不能据此使用测试服务标签重挑 checkpoint。

seed 42：

| 配置 | 验证事件 MRR | 验证事件平衡 CE | 测试事件 MRR |
| --- | ---: | ---: | ---: |
| LTFP | 1.000 | 0.044090 | 0.930556 |
| Raw | 1.000 | 0.040262 | 0.833333 |
| Deviation | 1.000 | 0.038765 | 0.805556 |

验证 MRR 原本为 1，CE 下降没有增加事件排名；测试 CE 则从 0.678023 升到 1.172238。连续损失选模并没有解决验证覆盖不足。

三个 currency CPU 事件（event_8 / event_9 / event_10），LTFP 均排第 1，Deviation 两个种子都降为第 6。虽然部分 checkout 事件得到改善，总体仍下降。

在基线排对的 144 个窗口中，Deviation 两个种子分别修坏 43 / 44 个，只修正原错误窗口 2 / 2 个。seed 42 中有 42 个受损窗口转而把 recommendation 排到第一。

这 42 个窗口中，currency 相对新竞争者的平均 logit margin 为 +0.535，修正后变成 -2.486；currency 的 latest 四指标偏差均值约 3.166，recommendation 约 0.205。局部强信号存在，但自由 MLP 仍可能输出反向修正。这是四指标均值比较，不代表逐指标都更高，也不能证明均值应无条件决定根因。

## TT fold 3：稀少根因的排序被改写

- 训练：route 10、train 7、auth 8、order 1、travel 13。
- 验证：route 4、travel 5。
- 测试：auth 8、train 3、order 2、travel 3。

order 只有一个训练根因事件，验证没有 order 根因。

| 事件 | LTFP rank | Dev seed 42 | Dev seed 43 |
| --- | ---: | ---: | ---: |
| order memory event_18 | 1 | 1 | 2 |
| order memory event_19 | 1 | 2 | 3 |
| auth delay event_6 | 3 | 5 | 5 |

基线排对的 94 个窗口中，Deviation 修坏 3 / 7 个，没有修正任何原错误窗口。主要竞争者是 travel，seed 43 也包括 auth。

seed 42 两个 order → travel 的受损窗口，order latest 偏差均值为 6.256，travel 为 0.307；原 margin +0.119，修正后 -1.070。seed 43 三个 order → auth 的窗口，原 margin 平均 +2.275，修正后 -1.509。错误并不局限于几乎打平的候选。

## 方法决策

本轮诊断到此结束，无需再增加图结构消融。

当前已明确两个问题：自由修正可能违背局部偏差证据；修正幅度没有约束，能够推翻原本正确的排名。验证服务覆盖不足使这些更新仍可能降低验证损失。后续模型应先定义局部证据如何贡献分数、允许怎样的排序修正，再设计满足这些性质的模块。

“基线低置信度才修正”还不能直接成为答案：OB 很多被修坏的正确窗口，基线 top 概率只有约 0.36，低置信度同样可能正确。需要使用训练/验证数据制定统一规则，不能根据这些测试案例为某折单独选回 epoch 0 或设置服务专属阈值。

保留正常原型作为候选局部证据；RCA 修正稳定前，系统预测主干继续保留。本轮没有 AP 改善证据，也尚未完成服务级—系统级协同。

这些是事后选取的两个退化折，足以指导下一次结构选择，但不足以证明所有部署都具有同一失败机制。模型偏向验证集常见特征模式是与结果一致的解释，具体输入的因果作用尚未通过干预证明。

## 复现与结果

- `scripts/audit_metric_bypass_failures.py`：重放四组数据/种子组合，统计服务覆盖、验证/测试损失、事件排名及逐窗口 margin。
- [完整事件与窗口结果](../artifacts/metric_bypass_failure_audit_2026-11-18.json)。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B \
  scripts/audit_metric_bypass_failures.py \
  --output /private/tmp/metric_bypass_new_failure_audit.json
```

使用已有 `/private/tmp/metric_bypass_seed42_20261118`、`/private/tmp/metric_bypass_seed43_20261118` checkpoint，原始数据标准化与历史基线一致性已核对。
