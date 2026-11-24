# OC-SID 独立模型设计（面向 SCI 期刊）

日期：2026-11-23

状态：统一模型已实现并跑通五折，结果见
`docs/2026-11-23-ocsid-unified-model-result.md`。
预测侧 AP 已追平/超过 LTFP；定位侧 OB 仍需接入 CPCA 融合。
LTFP 在本方案中只作为 baseline，不再作为被扩展的主干。

## 0. 结论

1. **采纳独立模型。** OC-SID 从随机初始化训练，拥有自己的编码器、预测头和归因头，
   不加载 LTFP 权重、不共享 LTFP 的 TCN/分钟图/RCA head。
2. **GPT 给的骨架方向是对的，但按原样实现有三个硬问题**，必须改：
   - operation 级标签没有现成定义（但 TT 的标签本身是 per-operation 的，可以导出）；
   - 5 个可学习模块 + 55/64 个事件，过拟合与跨种子不稳定几乎必然；
   - “把 source 表示换成正常原型再跑预测头”的 restoration，和已经失败的
     normal-restoration v0 是同一机制（Hit@1 只有 `0.31/0.56`），大概率重蹈覆辙。
3. **修正方案：** 保留 GPT 的 operation 级骨架和 source–impact 结构，
   把 restoration 换成结构反事实（CPCA，零训练），把 V1 的可学习模块压到 3 个，
   request-DAG GNN 放到 V2。这样既满足“独立模型”，又能在这份小数据上做统计。
4. **SCI 定位：** 贡献不是“又一个 SOTA”，而是
   “operation 级问题定义 + 结构反事实归因 + 严格的事件级评估与完整基线”。
   这在 Q2/Q3 的软件工程/系统期刊是站得住的；追 SOTA 反而会被 55 个事件打脸。

## 1. 数据现实决定架构

| 事实 | 数字 | 对架构的影响 |
|---|---|---|
| OB 只有 1 个入口 operation | `frontendservice / frontend` | operation 条件化在 OB 上退化为系统级；不能宣称两数据集都受益 |
| TT 每个 case 有约 15 个入口 operation | 单 case 实测 15 个 | operation 级建模只在 TT 有区分度；必须单独报告 |
| TT 标签本身就是 per-operation 定义 | `same-operation entry p95 degradation`，每 operation 自己的注入前基线 | `q_o` 可以导出，不需要发明新任务 |
| TT `cases.csv` 已存每 operation 的 baseline/threshold | `operation_thresholds_json` | 可直接生成 `y_{o,t}` |
| 事件数很少 | OB 55 / TT 64，每折 6–18 | 可学习模块必须少；主指标用事件级配对统计 |
| 预测被 persistence 支配 | persistence F1 `0.921/0.894` > LTFP `0.833/0.785` | 不要把“预测提升”当主卖点；主卖点是定位与机制 |

**直接推论：** GPT 的 `q_o` 路线在 TT 上是自洽的，在 OB 上必须退化为
`q = 系统风险`。论文里应把 OB 写成“单入口 operation 的退化情形”，
而不是假装 operation 级在两数据集都成立。

## 2. 模型 V1（先做这个）

### 2.1 数据层

每分钟生成两张表（复用 Gate A 的流式 trace 解析）：

**Operation 表**（每个入口 operation `o`，每分钟一行）：

- `count, p50, p95, p99, timeout_rate`
- `inclusive_p95, exclusive_p95, post_gap_p95, pre_gap_p95`
- `p95_slope`（过去 3 分钟）、`p95_ratio_to_baseline`
- `observed`（span 数是否足够）

**Service 表**（每个服务 `i`，每分钟一行）：

- 现有 metric/log 19 维特征（原 panel 可直接复用）；
- `local_p95`（自身处理时间）、`post_gap_p95`（作为被调用方的返回等待）；
- `inclusive_p95`、`coverage`（经过 i 的入口请求占比）、`slack_share`
  （i 在关键路径上的占比，定义见 CPCA 文档）；
- `path_mask`（该服务是否在 operation o 的路径上）。

归一化：只用训练折拟合；case 内滚动正常参考只使用 `t` 之前的分钟。

### 2.2 三个可学习模块（V1）

```text
Operation 表 ── Operation Temporal Encoder ── q_o（每个入口 operation 的下一分钟风险）
Service 表   ── Source Encoder ─┐
                                 ├─ Source–Impact Encoder ── a_i（初始根因分数）
路径/覆盖特征 ── Impact Encoder ─┘
Trace 结构   ── CPCA（零训练） ── Δ_i（正常恢复责任）
```

1. **Operation Temporal Encoder**：每个 operation 独立的 1 层 GRU 或 2 层因果 TCN，
   输入该 operation 的 5 分钟特征，输出 `h_o`。
2. **Operation Risk Head**：`q_o = sigmoid(MLP(h_o))`。
3. **Source–Impact Encoder**：
   - `s_i = MLP([local_dev, post_gap_dev, metric_dev, log_novelty])`
   - `e_i = MLP([inclusive_dev, coverage, slack_share, depth])`
   - `a_i = MLP([s_i, e_i, s_i ⊙ e_i])`

系统概率三选一，作为消融而不是拍脑袋：

| 聚合 | 公式 | 说明 |
|---|---|---|
| product | `p = 1 − ∏_o (1 − q_o)` | GPT 的方案；假设 operation 独立，可能高估 |
| max | `p = max_o q_o` | 最保守，适合“任一入口退化” |
| attention | `p = σ(Σ_o w_o · logit(q_o))`，`w = softmax(MLP(h_o))` | 多一个 pooling，参数量小 |

主表用 product 还是 attention，由验证集 AP 决定，且必须在论文里报告三种的结果。

### 2.3 归因头：CPCA 代替学习式 restoration

GPT 的 `Δ_i = q_o − q_o^{restore(i)}` 思路正确，但实现方式应改为结构反事实：

- 在请求的 span 树上，把服务 `i` 的 local 与进出它的 gap 替换为正常参考中位数；
- 递归重放，得到 `D^{cf}`，`Δ_i = D − D^{cf}`；
- 归一化为 `r_i = Σ Δ_i / Σ D`；
- 覆盖率/稳定性做解析式收缩。

它与 GPT 的区别只是“恢复谁、怎么算”：GPT 在**表示空间**恢复再跑预测头，
CPCA 在**结构时延**上恢复再重放。后者不需要训练，也不会因为预测头不依赖该输入而失效。
两者可以同时保留：`R_i = Σ_o q_o · (a_i,o + λ · r_i,o)`，`λ` 在验证集选，允许 0。

### 2.4 损失

```text
L = BCE(p, y_system)
  + α · BCE(q_o, y_o)                 # 仅 TT 有 operation 标签；OB 跳过
  + β · pairwise_rank(R, root)        # 仅正例窗口；事件等权
  + γ · BCE(a_i 聚合, y_system)       # 可选，防止 source-impact 头退化
```

- `α=β=0.5` 起步，只在验证集调一次；
- 根因排序损失按事件加权（每事件总权重相等），避免长事件主导；
- 不使用 LTFP 的任何预训练权重、冻结模块或旧预测。

### 2.5 V2（可选，V1 有效后再加）

Request DAG Encoder：对每条请求做 1–2 层有向消息传递，输入 span 的
`[service emb, operation emb, duration, exclusive, relative start, child count]`，
输出 `h_{i,r}`，再按 operation 聚合。V2 的增量必须单独做消融；
如果 V1 的结构特征已经吃掉大部分信号，V2 不值得加。

## 3. 与 GPT 方案的逐项差异

| GPT 设计 | 本方案 | 原因 |
|---|---|---|
| Span Encoder + Request DAG Encoder 作为主输入 | V1 用结构汇总特征，V2 再加 DAG GNN | 14M 请求上做 GNN 成本高，且结构特征已含机制信息 |
| Operation Temporal Encoder | 保留 | TT 有 15 个入口 operation，是真实粒度 |
| `q_o` 直接由系统标签训练 | 显式导出 `y_o`（TT），OB 退化为系统头 | 否则 `q_o` 没有监督，product 聚合无意义 |
| Source–Impact Encoder | 保留，但输入换成结构量 | 结构与机制对齐，参数更少 |
| 学习式 Normal-Restoration | 结构反事实 CPCA | 学习式 restoration 已失败（v0 Hit@1 0.31/0.56） |
| 5 个模块 | V1 三个模块 + 零训练归因 | 55/64 个事件，参数越多越不可信 |
| 独立模型 | 保留 | 这是正确的论文定位 |

## 4. 实验协议（SCI 版）

### 4.1 主表

| 模型 | 输入粒度 | 事件 Hit@1 | 事件 MRR | 窗口 AP | F1@0.5 |
|---|---|---:|---:|---:|---:|
| persistence | 分钟 | — | — | — | — |
| LTFP | 服务—分钟 | 0.807/0.894 | 0.878/0.932 | 0.919/0.846 | 0.837/0.771 |
| CIRCA / TraceRCA / MicroRank / TVDiag / Hades | 各自 | 需跑 | 需跑 | — | — |
| OCSID-V1 | operation—路径—分钟 | 目标 ≥ 基线 | 目标 +0.02 | 不低于 LTFP −0.01 | 同 |
| OCSID-V2 | + 请求 DAG | 消融 | 消融 | 消融 | 消融 |

所有数字用 5 folds × 3 seeds；RCA 用**事件级**，报逐折范围 + 配对 bootstrap 95% CI；
同时报每事件 win/loss，不只报均值。

### 4.2 必做消融

1. 去掉 `q_o`，只用系统头；
2. product / max / attention 三种聚合；
3. 去掉 CPCA，只留 `a_i`；
4. 去掉 source 或 impact 一支；
5. `λ = 0`（同 checkpoint 事后消融）；
6. 只用 metric / metric+trace / 全模态（对齐现有模态结论）。

### 4.3 分析

- 逐故障类型（6 类 × 2 数据集）：预期 CPU/disk 由 local 主导，delay 由 post-gap 主导，
  loss 较弱（实测 +4%~+9%）；这张表是机制成立与否的关键。
- 逐入口 operation（TT）：哪些 operation 上 `q_o` 有效。
- 首分钟 vs 持续分钟：诚实报告 onset 只有 10/20 个正例。
- 效率：训练时间、推理延迟、参数量的对比（期刊审稿人会看）。

## 5. 三个 Gate 与止损

### Gate 1：operation 标签自洽（1 周）

- 从 `cases.csv` 的 `operation_thresholds_json` + 原始 trace 生成 `y_{o,t}`；
- 检查 `1 − ∏(1−y_o)` 是否与现有 `failure_next_minute` 高度一致（应接近 1）；
- 若不一致，说明 operation 规则与系统规则不同源，`q_o` 路线降级为辅助头。

### Gate 2：CPCA 归因审计（1–2 周，不训练）

**已完成初版，见 `docs/2026-11-23-cpca-gate2-result.md`。**
结果：OB 事件 H@1 `0.891` / MRR `0.938`，显著优于 exclusive_z（CI `[+0.42,+0.61]`）；
TT `0.938/0.959`，优于但配对 CI 跨 0（TT 接近天花板）。判定为**有条件通过**。
剩余：完全对齐 Gate A 的 p95/robust-z 基线、OB covered-only 的 margin 分析。

### Gate 3：独立模型对比（2 周）

**预测侧第一版已完成，见 `docs/2026-11-23-ocsid-v1-prediction-baseline.md`。**
TT 五折 AP `0.777` / F1 `0.717`，LTFP 为 `0.846/0.771`；缺口集中在 fold 0（`0.383` vs `0.622`），
原因是缺 per-service metric/log 上下文与验证选模偏差。先补这两项，再判 Gate 3。

- OCSID-V1 与 LTFP 同协议、同数据、同五折、3 seeds；
- 通过线同上；若只在一个数据集有效，论文就只声称那个数据集，另一个写成边界分析。

**止损：** 若 Gate 2 或 Gate 3 不通过，停止加模块，转为
“严格任务定义 + 完整基线（10 specialist）+ 负结果与边界分析”的 measurement 论文。
这在 SCI 期刊同样可发，且比硬凑 SOTA 更稳。

## 6. 论文定位与选刊

### 6.1 贡献写法

1. **问题**：现有 RCA 在分钟级聚合上无法区分根因与症状；我们把它重新定义为
   operation 级的“可移除时延”估计问题。
2. **方法**：独立的 operation 级模型 + 结构反事实归因（CPCA），不依赖预训练主干。
3. **评估**：统一协议下 4 个框架 baseline + 10 个 specialist + persistence +
   first-visible；事件级配对统计；逐故障类型机制分析。

不要写“首次”“SOTA”“显著优于所有方法”；写“在什么条件下有效、在什么条件下退化”。

### 6.2 选刊（按难度）

| 档次 | 期刊 | 说明 |
|---|---|---|
| Q1，难 | JSS、IST、EMSE、TSE | 需要很强的评测与写作；55 个事件是硬伤，除非跨数据集/跨系统补强 |
| Q2，现实 | Journal of Cloud Computing、Cluster Computing、Computing、Software: Practice and Experience | 系统/软件方向，接受实证+方法 |
| Q3/快 | Applied Sciences、Electronics、Sensors、IEEE Access | 审稿快，但需要把工程完整性写足 |

以当前数据规模，**Q2 是合理目标**；Q1 需要额外补数据集（MicroSS/GAIA 只能做补充）
或补跨系统迁移实验。

### 6.3 审稿人最可能问的五个问题

1. 55 个事件，你的提升是否只是噪声？→ 配对 bootstrap + 每事件 win/loss + 3 seeds。
2. 为什么不用现成的 CIRCA/TraceRCA？→ 全部跑，同协议。
3. operation 级标签是不是你发明的？→ 引用 TT 原始标签规则，证明同源。
4. 归因是不是在拟合注入服务而不是用户影响？→ 报无影响 case 的源识别结果。
5. 计算成本？→ 报训练/推理时间。

## 7. 八周计划

| 周 | 任务 | 产出 |
|---|---|---|
| 1 | operation 标签生成 + Gate 1 | `y_{o,t}` 表、一致性报告 |
| 2 | operation/service 特征表（复用 trace 解析） | `artifacts/re2_{ob,tt}_ocsid_features_v1` |
| 3 | CPCA 归因审计（Gate 2） | 审计 JSON + 逐故障类型表 |
| 4 | OCSID-V1 三个模块 + 训练 | 五折 × 1 seed |
| 5 | 调参与消融（只调一次） | 消融表 |
| 6 | 3 seeds + 10 specialist 基线 | 主表 |
| 7 | V2（可选）+ 效率 + case study | 补充实验 |
| 8 | 写作 | 投稿稿 |

## 8. 现在就要做的三件事

1. **确认 TT 的 operation 标签能从 `cases.csv` + trace 复现**（Gate 1）。
   这是整个独立模型的地基；如果 `y_o` 不可导出，GPT 的 `q_o` 结构就要改成
   “operation 风险作为 attention 权重”，而不是概率。
2. **把 CPCA 的离线审计跑出来**（Gate 2）。它决定归因头是结构反事实还是学习式。
3. **不要先写模型代码。** 先有 Gate 1/2 的数字，再决定 V1 的输入和损失；
   否则又会回到“模型做完才发现数据里没有信号”的老路。

## 9. 与现有文档的关系

- `docs/2026-11-23-critical-path-counterfactual-design.md`：CPCA 归因头的完整定义、
  真实 trace 归属验证、可靠性收缩与风险表。本方案把它作为 OCSID 的归因头。
- `docs/2026-11-23-re2-data-driven-method-design.md`：数据审计与 Gate A 的原始结论。
- `scripts/audit_span_cost_signature.py`：§3.1.1 归属规则的复现脚本。
