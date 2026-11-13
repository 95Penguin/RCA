# 标签与评估协议

当前研究任务是用过去五个完整分钟的遥测预测下一分钟的**业务退化状态**，并对可归因故障事件输出根因服务排序。故障持续期间的下一分钟也属于正例；注入时间仅用于候选根因核对，不直接定义业务故障，也不作为模型输入。

## MicroSS / GAIA（7 月）

主实验使用 `entry_failure_v2.csv`：`webservice1/2` 合并的每分钟调用 span 至少 20 条，其中 5xx 比例达到 0.5，且该条件连续两分钟成立。离线事件起点回溯到首个高错误分钟；在线确认需要第二分钟结束。入口请求量不足的目标分钟标为未知并排除。原始面板包含 440,430 行（10 个服务）；当前 `available` 覆盖策略得到 41,942 个有效五分钟窗口。

根因标签是事件起点时唯一可靠、处于活动状态的注入服务。无法唯一匹配的事件保留故障预测标签，但不用于根因监督或评价。160 个故障事件中只有 15 个符合该规则。`entry_failure_v2.schema.json` 记录阈值、事件范围、排除的注入记录和标签覆盖。

固定主实验为事件隔离的时间顺序划分，输入和离线确认窗口不得跨划分边界；动态调用图只取输入分钟。训练、验证、测试有效窗口为 29,358／6,285／6,287，正例为 1,071／130／202。测试有 31 个带完整首分钟窗口的事件，其中 5 个有根因候选。配置与三种子 30 轮结果见 [三数据集基线](three-dataset-baselines.md)。

```bash
.venv/bin/python scripts/run_micross.py train \
  --data-dir data/processed/micross_mps_smoke_01 \
  --graph data/processed/micross_mps_smoke_01/service_adjacency_1m.npz \
  --labels data/processed/micross_mps_smoke_01/entry_failure_v2.csv \
  --output-dir artifacts/micross_example \
  --model gcn-gru --rca-method node-head --epochs 30 --device auto
```

`--model gru` 在同一数据与训练流程下关闭图卷积。`--coverage-policy complete` 要求所有服务资源指标齐全，是旧实验设定，不与主结果混合。论文式 0.999 错误率阈值仅作为敏感性检查；本研究固定的 0.5 阈值不是论文 SLA 数字。标签构建、阈值敏感性和源数据范围详见 [数据集协议](dataset-label-protocol.md)。

## RCAEval RE2-OB / RE2-TT

两个系统的入口 trace 缺少足够的 HTTP 5xx 业务响应证据，因此分别用入口操作的 p95 延迟相对注入前基线定义可观测退化。注入所在分钟不用于目标；连续两个满足阈值的完整分钟构成故障事件。注入服务只在业务退化可观测时作为候选根因。数据集各有 90 个官方注入案例；OB 有 55 个可评估事件，TT 有 64 个。两者采用部署组五折、三个种子；每个案例独立构建特征，标准化参数只用对应训练折拟合。具体阈值、覆盖条件和命令见 [数据集协议](dataset-label-protocol.md)，结果见 [RE2 多种子复核](re2-30ep-multiseed-and-tt-errors.md)。

## 报告方式

- 预测：完整测试窗口的 Precision、Recall、F1@0.5、AP；另报事件首分钟报警数。
- 定位：只在有候选根因的事件上计算 Hit@K/MRR，并明确窗口级与事后事件聚合；另报首分钟“报警且根因 Top-1”联合命中。
- 每个数据集单独报告样本量、正例量、事件数和根因覆盖，不把三个数据集的绝对分数混合平均。
- 当前分钟状态直接预测下一分钟是持续故障参照规则；它与模型在同一测试窗口比较。

三个数据集的业务标签定义、服务数量和切分单位不同，本文结果不直接等同于 InstantOps 论文公布的数值。
